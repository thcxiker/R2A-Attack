"""Surrogate training (Sec. 3.1, "Surrogate Router Training").

1. Observe the target on at most Q prompts of D_proxy (or read logged decisions).
2. Collect every open-source member's scores on the same prompts (free: the
   members are local models).
3. Fit alpha, W_l and beta so that the surrogate reproduces the target.

Everything is black-box: the target is only asked which model it selects
(through ``BlackBoxTarget``), at most Q times. The loss is a label-smoothed
cross-entropy on the observed decisions: the decision is turned into a
label-smoothed one-hot distribution (smoothing 0.2) and the loss is
0.1 * KL(label || surrogate) + 10 * CE(decision, smoothing 0.1); with
``decision_loss: ce`` plain cross-entropy is used instead.

Sources of decisions:

* live queries (default): Q random prompts of D_proxy are sent to the target;
* ``decision_logs``: Q random previously logged decisions (e.g. OpenRouter);
  the target is not queried.

Prompts that occur in any evaluation set (or in D_suffix) are never used.
"""

import copy
import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from r2a.blackbox import BlackBoxTarget, load_decision_logs
from r2a.surrogate.adapter import LowRankSemanticAdapter, build_router_stack
from r2a.utils import extract_text, query_source

logger = logging.getLogger(__name__)


@dataclass
class SurrogateConfig:
    query_budget: int = 120
    decision_loss: str = "blackbox_hybrid"  # blackbox_hybrid | ce
    label_smoothing: float = 0.2
    target_pool: str = "observed"       # observed: models seen in the decisions | declared: the router's candidate list
    decision_logs: Optional[List[str]] = None  # train from logged decisions instead of querying the target
    val_ratio: float = 0.2
    epochs: int = 20
    batch_size: int = 32
    lr: float = 0.03
    optimizer: str = "adamw"            # adamw | adam
    rank: int = 16
    embedding_dim: int = 384
    text_embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    patience: int = 10


def collect_decisions(
    target: Optional[BlackBoxTarget],
    pool: List[Dict],
    cfg: SurrogateConfig,
    seed: int,
    exclude: Optional[Set[str]] = None,
):
    """Returns (samples with the ``decision`` model name, target pool, #target queries)."""
    rng = random.Random(seed)
    exclude = exclude or set()
    if cfg.decision_logs:
        samples = []
        for s in load_decision_logs(cfg.decision_logs):
            if s["prompt"].strip() in exclude or s.get("origin_query", "").strip() in exclude:
                continue
            samples.append({k: s[k] for k in ("prompt", "source", "decision")})
        logger.info("%d logged decisions left after removing evaluation prompts", len(samples))
        rng.shuffle(samples)
        samples = samples[: cfg.query_budget]
        n_queries = 0
    else:
        items = [it for it in pool if extract_text(it["question"]).strip() not in exclude]
        rng.shuffle(items)
        items = items[: cfg.query_budget]
        samples = []
        for item in tqdm(items, desc="Querying target (decisions only)", leave=False, disable=None):
            prompt, source = extract_text(item["question"]), query_source(item)
            samples.append({"prompt": prompt, "source": source, "decision": target.decide(prompt, "", type=source)})
        n_queries = target.queries
    if cfg.target_pool == "declared":
        if target is None:
            raise ValueError("target_pool: declared needs a target router; use observed with decision_logs")
        names = target.candidate_pool()
        unknown = {s["decision"] for s in samples} - set(names)
        if unknown:
            raise ValueError(f"decisions outside the declared pool: {sorted(unknown)[:5]}")
    elif cfg.target_pool == "observed":
        names = sorted({s["decision"] for s in samples})
    else:
        raise ValueError(f"unknown target_pool '{cfg.target_pool}'")
    index = {n: i for i, n in enumerate(names)}
    for s in samples:
        s["label"] = index[s["decision"]]
    return samples, names, n_queries


def _loss(pred: torch.Tensor, labels: torch.Tensor, cfg: SurrogateConfig):
    if cfg.decision_loss == "ce":
        return F.cross_entropy(pred, labels, label_smoothing=cfg.label_smoothing)
    if cfg.decision_loss == "blackbox_hybrid":
        n = pred.shape[-1]
        smooth = cfg.label_smoothing if n > 1 else 0.0
        soft = torch.full_like(pred, smooth / max(n - 1, 1))
        soft.scatter_(1, labels.view(-1, 1), 1.0 - smooth)
        kl = F.kl_div(F.log_softmax(pred, dim=-1), soft, reduction="batchmean")
        return 0.1 * kl + 10.0 * F.cross_entropy(pred, labels, label_smoothing=0.1)
    raise ValueError(f"unknown decision_loss '{cfg.decision_loss}'")


def train_surrogate(
    members: List,
    samples: List[Dict],
    unified_model_names: List[str],
    target_model_names: List[str],
    cfg: SurrogateConfig,
    device,
    seed: int,
) -> Tuple[LowRankSemanticAdapter, Dict]:
    torch.manual_seed(seed)
    adapter = LowRankSemanticAdapter(
        unified_model_names=unified_model_names,
        target_model_names=target_model_names,
        text_embedder_name=cfg.text_embedder,
        device=device,
        num_routers=len(members),
        embedding_dim=cfg.embedding_dim,
        rank=cfg.rank,
    ).to(device)

    # The members are frozen and the queries carry no suffix, so their scores
    # are computed once and reused in every epoch.
    logger.info("Collecting member-router scores on %d queries...", len(samples))
    stacks, masks = [], []
    for s in tqdm(samples, desc="Member routers", leave=False, disable=None):
        a, m = build_router_stack([s["prompt"]], members, unified_model_names, device, [s["source"]])
        stacks.append(a)
        masks.append(m)
    stack, mask = torch.cat(stacks), torch.cat(masks)
    labels = torch.tensor([s["label"] for s in samples], device=device)
    with torch.no_grad():
        embeddings = adapter.embedder.encode_texts([s["prompt"] for s in samples])

    n_val = int(round(len(samples) * cfg.val_ratio))
    n_train = len(samples) - n_val
    tr, va = torch.arange(n_train), torch.arange(n_train, len(samples))

    params = [p for p in adapter.parameters() if p.requires_grad]
    opt_cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
    optimizer = opt_cls(params, lr=cfg.lr)

    def forward(idx):
        return adapter.combine(stack[idx], mask[idx], adapter.lowrank(embeddings[idx]))

    def loss_on(idx, pred):
        return _loss(pred, labels[idx], cfg)

    def evaluate(idx):
        adapter.eval()
        with torch.no_grad():
            pred = forward(idx)
            loss = loss_on(idx, pred).item()
            acc = (pred.argmax(-1) == labels[idx]).float().mean().item()
        adapter.train()
        return loss, acc

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    gen = torch.Generator().manual_seed(seed)
    adapter.train()
    for epoch in range(1, cfg.epochs + 1):
        perm = tr[torch.randperm(n_train, generator=gen)]
        total = 0.0
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            loss = loss_on(idx, forward(idx))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history["train_loss"].append(total / max(1, n_train))
        if n_val == 0:
            logger.info("Epoch %d/%d | train %.4f", epoch, cfg.epochs, history["train_loss"][-1])
            continue
        val_loss, val_acc = evaluate(va)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        logger.info(
            "Epoch %d/%d | train %.4f | val %.4f | val top-1 agreement %.1f%%",
            epoch, cfg.epochs, history["train_loss"][-1], val_loss, 100 * val_acc,
        )
        if val_loss < best_loss:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(adapter.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break
    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.eval()

    summary = {
        "best_epoch": best_epoch,
        "train_top1_agreement": evaluate(tr)[1],
        "val_top1_agreement": evaluate(va)[1] if n_val else None,
        "router_weights": dict(zip(["lowrank"] + [m.name for m in members], adapter.get_router_weights().tolist())),
        "target_pool": list(target_model_names),
        "label_distribution": {target_model_names[k]: v for k, v in Counter(labels.tolist()).items()},
        "history": history,
    }
    return adapter, summary
