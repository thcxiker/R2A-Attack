"""Universal suffix optimization with incremental query activation (Algorithm 1).

Queries of D_suffix that the surrogate already routes to a strong model are
dropped; the remaining m queries are activated one at a time. Each of the T
iterations takes one GCG step on the m_c active queries (summed gradients,
candidates scored by their mean loss over the active queries); once the suffix
succeeds on all m_c active queries the next query is activated, and the
optimization stops early once it succeeds on all m queries. Only the surrogate
is used; the target router is never queried.
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from r2a.attack.gcg import EnsembleGCG
from r2a.utils import extract_text, load_json, query_source, save_json

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    init_suffix: str = "! ! ! ! ! ! ! ! ! !"
    max_total_steps: int = 3000         # T
    max_train_samples: Optional[int] = None  # cap on the number of queries m (smoke tests)
    log_every: int = 10                 # save progress every n steps


def _queries(samples: List[Dict]):
    return [(extract_text(s.get("question", "")), query_source(s)) for s in samples]


def _all_succeed(gcg: EnsembleGCG, queries, suffix: str) -> int:
    return sum(gcg.outcome(q, suffix, t)["strong"] for q, t in queries)


def _optimize(gcg: EnsembleGCG, active: List[Dict], cfg: StageConfig, progress_path: Path, resume: bool) -> Dict:
    queries = _queries(active)
    m = len(queries)
    suffix, m_c, step, activations = cfg.init_suffix, 1, 0, []
    if resume and progress_path.exists():
        state = load_json(str(progress_path))
        suffix, m_c, step, activations = state["suffix"], state["m_c"], state["step"], state["activations"]
        logger.info("Resuming at step %d (m_c=%d) with suffix %r", step, m_c, suffix)

    def save():
        save_json({"suffix": suffix, "m_c": m_c, "step": step, "activations": activations}, str(progress_path))

    done = m == 0
    t0 = time.time()
    while not done and step < cfg.max_total_steps:
        batch = queries[:m_c]
        suffix, loss = gcg.joint_step(batch, suffix)
        step += 1
        n_ok = _all_succeed(gcg, batch, suffix)
        logger.info("Step %d | m_c %d/%d | loss %.4f | success %d/%d | %r", step, m_c, m, loss, n_ok, m_c, suffix)
        if n_ok == m_c:
            activations.append({"step": step, "m_c": m_c, "suffix": suffix, "seconds": round(time.time() - t0, 1)})
            if m_c < m:
                m_c += 1
            else:
                done = True
            save()
        elif step % cfg.log_every == 0:
            save()
    save()
    if done:
        logger.info("Suffix succeeds on all %d queries after %d steps.", m, step)
    else:
        logger.info("Reached T=%d steps with %d/%d queries active.", cfg.max_total_steps, m_c, m)
    return {
        "suffix": suffix,
        "num_active_queries": m,
        "activated_queries": m_c,
        "total_gcg_steps": step,
        "succeeded_on_all": done,
        "activations": activations,
    }


def optimize_universal_suffix(
    gcg: EnsembleGCG,
    train_samples: List[Dict],
    cfg: StageConfig,
    output_dir: str,
    resume: bool = False,
) -> Dict:
    out = Path(output_dir)
    logger.info("Filtering D_suffix with the surrogate (%d queries)...", len(train_samples))
    active = [
        s for s in train_samples
        if not gcg.outcome(extract_text(s.get("question", "")), "", query_source(s))["strong"]
    ]
    logger.info("%d / %d queries are routed to weak models by the surrogate", len(active), len(train_samples))
    if cfg.max_train_samples is not None:
        active = active[: cfg.max_train_samples]
    result = _optimize(gcg, active, cfg, out / "suffix_progress.json", resume)
    result["stage_config"] = asdict(cfg)
    save_json(result, str(out / "trained_suffix.json"))
    return result
