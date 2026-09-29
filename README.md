# Route to Rome Attack (R2A)

Code for **"Route to Rome Attack: Directing LLM Routers to Expensive Models via
Adversarial Suffix Optimization"**.

Cost-aware LLM routers send easy queries to cheap models and hard queries to
expensive ones. R2A learns a single *universal* adversarial suffix that, when
appended to a query, makes a **black-box** router send it to an expensive
(strong) model. The attacker only observes the router's decisions and may query
it at most Q = 120 times.

R2A has two stages:

1. **Hybrid ensemble surrogate router** (Eq. 3-5). Open-source routers
   (RouteLLM-BERT, RouteLLM-Causal, P2L, GraphRouter, RouterDC) are combined
   with a trainable low-rank router z_l = E(q) W_l^1 W_l^2 on MiniLM
   embeddings. Each open-source router's standardized scores are zero-padded
   into the union pool and mapped onto the target's pool by a trainable W_o;
   all K + 1 routers are mixed with weights alpha = softmax(.). The surrogate is
   fit to the target's decisions on Q queries.
2. **Suffix optimization** (Algorithm 1). A GCG-style search on the surrogate.
   Every member provides token gradients through its own tokenizer; per-router
   scores over a shared single-token vocabulary are min-max normalized and
   aggregated into top-k candidates. Each step optimizes the loss summed over
   the active queries, and the next query is activated once the suffix works
   on all active ones.

The attack success rate (ASR) is the fraction of evaluation queries that the
target routes to a strong model once the suffix is appended.

### Black-box access

Every experiment in this repository is black-box: the code never reads the
target router's scores, gradients or parameters, and the target is never part
of the surrogate ensemble. All access goes through
`r2a/blackbox.py:BlackBoxTarget`, which returns only the name of the selected
model and counts queries:

* **Surrogate training** sends at most `surrogate.query_budget` (Q = 120)
  prompts to the target and raises an error on the (Q+1)-th query. The labels
  are the selected model names. Alternatively, `surrogate.decision_logs`
  trains from logged decisions without querying the target at all (OpenRouter).
* **Suffix optimization** uses only the surrogate; the target is not queried.
* **Evaluation** counts how often the selected model is a strong model.
* **Data**: the query sets were sampled uniformly at random from the source
  benchmarks, without consulting any router. Prompts of D_suffix and of the
  evaluation sets are removed from the surrogate's training prompts (also from
  decision logs).

`tests/test_core.py` checks these properties (decision-only data flow, query
budget, held-out prompts, no router-based data selection).

## Installation

Tested with Python 3.10, PyTorch 2.1 (CUDA 12.x) and 2 x RTX A6000 (48 GB).

```bash
conda create -n r2a python=3.10 -y && conda activate r2a
pip install -r requirements.txt
pip install -e .
bash scripts/setup_p2l.sh      # fetches the P2L model code (not redistributed here)
```

`torch-geometric` (for GraphRouter) may need the wheel that matches your
PyTorch/CUDA build; see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Model assets

All routers except RouterDC load public checkpoints from the Hugging Face Hub.
They are downloaded on first use, or ahead of time with

```bash
huggingface-cli login                       # Meta-Llama-3-8B (RouteLLM-Causal tokenizer) is gated
bash scripts/download_models.sh             # everything
bash scripts/download_models.sh small       # only what configs/smoke.yaml needs
```

| Router | Checkpoint |
|---|---|
| RouteLLM-BERT / Causal / MF | `routellm/{bert,causal_llm,mf}_gpt4_augmented` |
| P2L | `lmarena-ai/p2l-7b-grk-02222025`, `lmarena-ai/p2l-1.5b-bt-01132025`, `lmarena-ai/p2l-0.5b-bt-01132025` |
| GraphRouter | shipped in `r2a/third_party/graphrouter/` (trained on the GraphRouter data) |
| RouterDC | **not public**: train it with the [official code](https://github.com/shuhao02/RouterDC) and put the weights at `checkpoints/routerdc/best_model.pth` |
| Lightweight router encoder | `sentence-transformers/all-MiniLM-L6-v2` |

RouteLLM-MF embeds prompts with the OpenAI embeddings API, so the
`routellm_mf` target needs `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`).
Local model directories can be used instead of Hub ids by overriding
`routers.<name>.args.model_path` (or `checkpoint_path`).

## Quick start

```bash
# 1) Check a config (no model is loaded)
python -m r2a check --config configs/paper/routellm_bert.yaml

# 2) End-to-end smoke test on tiny budgets (a few minutes on one GPU)
python -m r2a run --config configs/smoke.yaml

# 3) Full attack on one target (paper setting)
python -m r2a run --config configs/paper/routellm_bert.yaml
```

`run` executes the three stages; they can also be run separately:

```bash
python -m r2a surrogate --config CFG           # query the target, train the surrogate
python -m r2a attack    --config CFG           # optimize the suffix (add --resume to continue)
python -m r2a evaluate  --config CFG --suffix "mine=<any suffix>"
```

Any config value can be overridden from the command line, e.g.
`--set attack.max_total_steps=500 --set experiment.device=cuda:1`.

Results are written to `outputs/<experiment.name>/`:

| File | Content |
|---|---|
| `surrogate.pt`, `surrogate_summary.json` | surrogate weights, target queries used, agreement with the target, learned router weights |
| `suffix_progress.json`, `trained_suffix.json` | suffix after every stage (used by `--resume`) and the final universal suffix |
| `eval_report.json`, `eval_summary.md` | ASR of `clean`, the CoT baseline and the R2A suffix, per dataset and per source |
| `config.resolved.yaml`, `run.log` | the exact config and the log |

## Experiments

`configs/paper/` contains one config per target router. As in the paper, the
target is removed from the five-router pool before surrogate training:

| Config | Target | Ensemble members |
|---|---|---|
| `paper/routellm_bert.yaml` | RouteLLM-BERT | Causal, P2L-7B, GraphRouter, RouterDC |
| `paper/graphrouter.yaml` | GraphRouter | BERT, Causal, P2L-7B, RouterDC |
| `paper/p2l_7b.yaml` | P2L-7B | BERT, Causal, GraphRouter, RouterDC |
| `paper/routerdc.yaml` | RouterDC | BERT, Causal, P2L-7B, GraphRouter |
| `paper/routellm_mf.yaml` | RouteLLM-MF | all five |
| `paper/openrouter.yaml` | OpenRouter (commercial) | all five; trained from logged decisions |

For OpenRouter the surrogate is trained from logged routing decisions (the
`actual_model` OpenRouter reports for each prompt) placed under
`data/openrouter_logs/`; see `load_decision_logs` in `r2a/blackbox.py` for the
format. Evaluating a suffix on OpenRouter requires live API calls and is not
part of this repository.

Data (`data/`):

* `proxy_pool.json`: D_proxy, 5,387 MMLU / GSM8K / MT-Bench-101 prompts from
  which the Q queries to the target are drawn (`scripts/build_proxy_pool.py`).
  It shares no prompt with the evaluation sets.
* `in_distribution.json`: 285 MMLU-Pro, 215 GSM8K and all 80 MT-Bench queries,
  sampled at random (`scripts/build_eval_sets.py`) and split 70/30 (seed 42)
  into D_suffix (suffix optimization) and D_eval (in-distribution evaluation).
  `in_distribution_2.json` and `in_distribution_3.json` are two further random
  samples (`--set data.suffix_set=data/in_distribution_2.json`).
* `ood.json`: SimpleQA, ArenaHard and RouterArena queries, never used for training.

Main hyperparameters (`configs/base.yaml`):

| Stage | Parameter | Value |
|---|---|---|
| surrogate | query budget Q / rank r / epochs / lr / batch / optimizer | 120 / 16 / 20 / 0.03 / 32 / AdamW |
| attack | iterations T / candidates per step B / top-k / max suffix tokens | 3000 / 64 / 256 / 30 |
| attack | initial suffix | `! ! ! ! ! ! ! ! ! !` |

Each GCG step evaluates B = 64 candidates on every active query with every
ensemble member, so a step gets slower as more queries are activated. `T =
3000` is an upper bound; optimization stops once the suffix succeeds on every
query of D_suffix.

GPU placement is set per router (`routers.<name>.args.device`, and
`device_map` for RouteLLM-Causal). The default places RouteLLM-Causal on
`cuda:0` and P2L-7B on `cuda:1`, which fits the full ensemble (including
back-propagation through both 7-8B models) on 2 x 48 GB. On a single GPU, set
`--set routers.p2l_7b.args.device=cuda:0` and use shorter queries or a smaller
ensemble.

## Repository layout

```
r2a/
  cli.py, pipeline.py, config.py      entry points and orchestration
  routers/                            router adapters: scores + embedding-level forward for gradients
  surrogate/adapter.py, train.py      hybrid ensemble surrogate and its training
  attack/vstar.py, gcg.py, losses.py  candidate construction, ensemble GCG, attack losses
  attack/optimize.py                  universal suffix optimization with incremental query activation
  evaluate.py, classifier.py          ASR and the strong/weak model partition
  third_party/                        RouteLLM, P2L and GraphRouter model code (modified, see below)
configs/                              experiment configs, strong/weak partition, RouteLLM thresholds, P2L costs
data/                                 proxy pool and evaluation sets
scripts/                              model download, proxy-pool construction
tests/                                CPU unit tests (python -m pytest tests)
```

## Third-party code

`r2a/third_party/` contains modified code from
[RouteLLM](https://github.com/lm-sys/RouteLLM) (Apache-2.0) and
[GraphRouter](https://github.com/ulab-uiuc/GraphRouter) (MIT, including its
trained weights); their licenses are included next to the code. The changes
add embedding-level forward passes for gradient computation and remove
training-only code. [Prompt-to-Leaderboard](https://github.com/lmarena/p2l)
has no license file and is therefore not redistributed:
`scripts/setup_p2l.sh` clones it into `external/p2l` (or set `R2A_P2L_PATH`).
The RouterDC module in `r2a/routers/routerdc.py` follows
[RouterDC](https://github.com/shuhao02/RouterDC).

## Responsible use

This code is released to support research on the robustness of LLM routing.
Do not use it against services you do not own or have permission to test.

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
