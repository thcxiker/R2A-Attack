"""Fast CPU tests that need no model downloads: python -m pytest tests -q"""

import glob
import json
import random
from pathlib import Path

import pytest
import torch

from r2a.attack.losses import create_loss
from r2a.attack.optimize import StageConfig, optimize_universal_suffix
from r2a.attack.vstar import VStarBuilder
from r2a.classifier import ModelClassifier, routing_outcome
from r2a.config import REPO_ROOT, load_config, validate_config
from r2a.surrogate.adapter import LowRankSemanticAdapter
from r2a.utils import split_suffix_set

CLASSIFICATION = str(REPO_ROOT / "configs" / "model_classification.yaml")


# ---------------------------------------------------------------- configs
@pytest.mark.parametrize("path", sorted(glob.glob(str(REPO_ROOT / "configs" / "*" / "[a-z]*.yaml"))) + [str(REPO_ROOT / "configs" / "smoke.yaml")])
def test_shipped_configs_are_valid(path):
    cfg = load_config(path)
    logs = cfg.get("surrogate", {}).get("decision_logs") or []
    if logs and not all((REPO_ROOT / p).exists() for p in logs):
        pytest.skip("decision logs are not distributed with the repository")
    assert validate_config(cfg) == []
    assert cfg["target"]["router"] not in cfg["ensemble"]["members"]


def test_config_inheritance_and_overrides():
    cfg = load_config(str(REPO_ROOT / "configs" / "smoke.yaml"), ["attack.topk=7"])
    assert cfg["attack"]["search_width"] == 8                          # from smoke.yaml
    assert cfg["attack"]["loss"] == "strong_promotion"                 # from base.yaml
    assert cfg["attack"]["topk"] == 7


def test_validate_rejects_target_in_ensemble():
    cfg = load_config(str(REPO_ROOT / "configs" / "paper" / "routellm_bert.yaml"), ["ensemble.members=[routellm_bert]"])
    assert any("also an ensemble member" in e for e in validate_config(cfg))


# ---------------------------------------------------------------- strong / weak
def test_partition_of_router_pools():
    clf = ModelClassifier(CLASSIFICATION)
    assert clf.classify_list(["gpt-4", "mixtral-8x7b"]) == (["gpt-4"], ["mixtral-8x7b"])
    strong, weak = clf.classify_list(["LLaMA-3 (70b)", "LLaMA-3 (8b)", "Mixtral-8x7B", "Mistral-7b"])
    assert strong == ["LLaMA-3 (70b)", "Mixtral-8x7B"] and weak == ["LLaMA-3 (8b)", "Mistral-7b"]


def test_routing_outcome_argmax_and_ties():
    clf = ModelClassifier(CLASSIFICATION)
    pool = ["gpt-4", "mixtral-8x7b"]
    assert routing_outcome(torch.tensor([0.1, 0.0]), pool, clf)["strong"]
    assert not routing_outcome(torch.tensor([-0.1, 0.0]), pool, clf)["strong"]
    assert routing_outcome(torch.tensor([0.0, 0.0]), pool, clf)["strong"]      # ties count as strong


# ---------------------------------------------------------------- losses
def test_strong_promotion_loss():
    loss = create_loss("strong_promotion", [0], [1])
    assert loss(torch.tensor([0.0, 0.0])).item() == pytest.approx(0.5)
    logits = torch.tensor([0.0, 1.0], requires_grad=True)
    loss(logits).backward()
    assert logits.grad[0] < 0 < logits.grad[1]      # descent raises the strong logit


# ---------------------------------------------------------------- V*
class _Tok:
    def __init__(self, words, offset):
        self.vocab = {w: i + offset for i, w in enumerate(words)}
        self.inv = {i: w for w, i in self.vocab.items()}
        self.all_special_ids = []

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[w] for w in text.split()]


class _Enc:
    def __init__(self, name, words, offset, seed):
        self.name, self.tok = name, _Tok(words, offset)
        # orthonormal embeddings (randomly rotated) so the planted word is the unique best candidate
        q, _ = torch.linalg.qr(torch.randn(len(words), len(words), generator=torch.Generator().manual_seed(seed)))
        self.E = torch.cat([torch.zeros(offset, len(words)), q])

    def get_tokenizer(self):
        return self.tok

    def embed_matrix(self):
        return self.E


def test_vstar_returns_the_best_scoring_word():
    words = ["zeta", "alpha", "mu", "beta", "omega", "gamma", "kappa", "delta"]
    # ids are *not* in alphabetical order, as in real vocabularies
    encs = [_Enc("a", words, 0, 0), _Enc("b", list(reversed(words)), 100, 1)]
    vb = VStarBuilder(encs, k=1, device="cpu")
    hits = 0
    for w in words:
        row = vb.common_words.index(w)
        grads = [-vb.embedding_pools[0][row] * 100, -vb.embedding_pools[1][row] * 100]
        top = vb.find_topk_substitutions(grads)[0]
        hits += encs[0].tok.inv[top] == w
    assert hits == len(words)


# ---------------------------------------------------------------- surrogate combination
def _bare_adapter(num_routers, U, L):
    adapter = LowRankSemanticAdapter.__new__(LowRankSemanticAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.num_routers = num_routers
    adapter.router_weights = torch.nn.Parameter(torch.zeros(num_routers + 1))
    adapter.projection_matrix = torch.nn.Parameter(torch.eye(U, L))
    return adapter


def _z(x):
    return (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-6)


def test_paper_surrogate_is_eq5():
    """y = alpha_0 z_l + sum_k alpha_k W_o z_uni^(k), alpha = softmax over K + 1 routers."""
    adapter = _bare_adapter(2, 3, 3)
    stack = torch.tensor([[[1.0, 2.0, 3.0], [5.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, True], [False, False, False]]])   # second router has no candidate in M_uni
    z_l = torch.tensor([[0.5, -0.5, 1.0]])
    out = adapter.combine(stack, mask, z_l)
    assert torch.allclose(out[0], 0.5 * z_l[0] + 0.5 * _z(stack[0, 0]), atol=1e-5)
    assert adapter.projection_matrix.requires_grad        # W_o is trained
    with pytest.raises(ValueError):
        adapter.combine(torch.zeros(1, 3, 3), torch.ones(1, 3, 3, dtype=torch.bool), z_l)


# ---------------------------------------------------------------- data split
def test_split_is_seeded_shuffle():
    items = list(range(100))
    random.seed(42)
    ref = items[:]
    random.shuffle(ref)
    tr, te = split_suffix_set(items, 0.7, 42)
    assert tr == ref[:70] and te == ref[70:]


def test_data_splits_are_disjoint():
    def texts(path):
        out = set()
        for x in json.load(open(path)):
            q = x["question"]
            out.add((" ".join(q) if isinstance(q, list) else q).strip())
        return out

    pool = texts(REPO_ROOT / "data" / "proxy_pool.json")
    for name in ("in_distribution.json", "ood.json"):
        assert not pool & texts(REPO_ROOT / "data" / name)


# ---------------------------------------------------------------- staged optimization
class _FakeGCG:
    """Query i succeeds once the suffix contains at least i+1 '#' characters."""

    def outcome(self, q, suffix, task):
        return {"strong": suffix.count("#") > int(q)}

    def run(self, q, suffix, n, task):
        class R:
            best_string = suffix + "#"
        return R()


def test_algorithm1_activation_and_budget(tmp_path):
    class Joint(_FakeGCG):
        def joint_step(self, batch, suffix):
            return suffix + "#", 0.0

    samples = [{"question": str(i), "source": "mmlu"} for i in range(4)]
    res = optimize_universal_suffix(Joint(), samples, StageConfig(init_suffix="x", max_total_steps=100), str(tmp_path))
    # one query is activated per successful step; stops once all 4 succeed
    assert res["succeeded_on_all"] and res["total_gcg_steps"] == 4
    assert [a["m_c"] for a in res["activations"]] == [1, 2, 3, 4]
    res = optimize_universal_suffix(Joint(), samples, StageConfig(init_suffix="x", max_total_steps=2), str(tmp_path / "b"))
    assert not res["succeeded_on_all"] and res["total_gcg_steps"] == 2 and res["activated_queries"] == 3


# ---------------------------------------------------------------- black-box guarantees
class _SpyRouter:
    """Target router that records every call; its scores must never leave BlackBoxTarget."""

    name = "spy"

    def __init__(self):
        self.calls = 0

    def get_model_list(self):
        return ["gpt-4", "mixtral-8x7b"]

    def route(self, prompt, suffix="", type=None):
        self.calls += 1
        return torch.tensor([0.3 if "hard" in prompt + suffix else -0.3, 0.0])


def test_decision_labels_are_black_box_and_within_budget():
    from r2a.blackbox import BlackBoxTarget, QueryBudgetExceeded
    from r2a.surrogate.train import SurrogateConfig, collect_decisions

    pool = [{"question": f"{'hard' if i % 3 == 0 else 'easy'} question {i}", "source": "mmlu"} for i in range(500)]
    spy = _SpyRouter()
    target = BlackBoxTarget(spy, budget=120)
    samples, names, n = collect_decisions(target, pool, SurrogateConfig(query_budget=120), seed=0)
    assert spy.calls == n == 120 == len(samples)
    assert all(set(s) == {"prompt", "source", "decision", "label"} for s in samples)   # names only, no scores
    assert names == ["gpt-4", "mixtral-8x7b"]
    with pytest.raises(QueryBudgetExceeded):
        target.decide("one more")


def test_decision_logs_need_no_target(tmp_path):
    from r2a.surrogate.train import SurrogateConfig, collect_decisions

    log = {"dataset_name": "hle", "records": [
        {"prompt": f"q{i}", "extra_fields": {"actual_model": "openai/gpt-5" if i % 2 else "qwen/qwen3-14b"}} for i in range(10)
    ] + [{"origin_query": "q-str", "extra_fields": "{'actual_model': 'openai/gpt-5'}"}]}
    path = tmp_path / "log.json"
    path.write_text(json.dumps(log))
    cfg = SurrogateConfig(query_budget=6, decision_logs=[str(path)])
    samples, names, n = collect_decisions(None, [], cfg, seed=1, exclude={"q0", "q1"})
    assert n == 0 and len(samples) == 6
    assert names == ["openai/gpt-5", "qwen/qwen3-14b"]
    assert {s["source"] for s in samples} == {"hle"}
    assert not {"q0", "q1"} & {s["prompt"] for s in samples}      # held-out prompts are dropped


def test_evaluation_uses_decisions_only():
    from r2a.blackbox import BlackBoxTarget
    from r2a.evaluate import evaluate_suffixes

    spy = _SpyRouter()
    data = {"toy": [{"question": "easy one", "source": "mmlu"}, {"question": "easy two", "source": "mmlu"}]}
    rep = evaluate_suffixes(BlackBoxTarget(spy), ModelClassifier(CLASSIFICATION), data, {"s": " hard"})
    assert rep["toy"]["clean"]["asr"] == 0.0 and rep["toy"]["s"]["asr"] == 1.0
    assert set(rep["toy"]["s"]["records"][0]) == {"question_id", "source", "decision", "strong"}


def test_label_smoothed_decision_loss():
    """0.1 * KL(label-smoothed one-hot || pred) + 10 * CE(smoothing 0.1)."""
    import torch.nn.functional as F
    from r2a.surrogate.train import SurrogateConfig, _loss

    torch.manual_seed(0)
    pred, labels = torch.randn(5, 4), torch.tensor([0, 1, 2, 3, 1])
    probs = torch.full((5, 4), 0.2 / 3)
    probs[torch.arange(5), labels] = 0.8
    target_logits = torch.log(probs + 1e-9)
    ref = 0.1 * F.kl_div(F.log_softmax(pred, -1), F.softmax(target_logits, -1), reduction="batchmean") \
        + 10.0 * F.cross_entropy(pred, target_logits.argmax(-1), label_smoothing=0.1)
    assert _loss(pred, labels, SurrogateConfig()).item() == pytest.approx(ref.item(), rel=1e-5)


def test_target_cannot_be_an_ensemble_member():
    base = str(REPO_ROOT / "configs" / "paper" / "routellm_bert.yaml")
    assert any("ensemble member" in e for e in validate_config(load_config(base, ["ensemble.members=[routellm_bert]"])))


def test_live_queries_skip_held_out_prompts():
    from r2a.blackbox import BlackBoxTarget
    from r2a.surrogate.train import SurrogateConfig, collect_decisions

    pool = [{"question": f"q{i}", "source": "mmlu"} for i in range(10)]
    samples, _, n = collect_decisions(BlackBoxTarget(_SpyRouter(), budget=5), pool, SurrogateConfig(query_budget=5),
                                      seed=0, exclude={f"q{i}" for i in range(5)})
    assert n == 5 and {s["prompt"] for s in samples} == {f"q{i}" for i in range(5, 10)}


def test_query_sets_were_not_selected_with_a_router():
    for name in ("in_distribution.json", "in_distribution_2.json", "in_distribution_3.json"):
        for x in json.load(open(REPO_ROOT / "data" / name)):
            assert x["selection_type"] == "random"
            assert not any("prob" in k for k in x)
