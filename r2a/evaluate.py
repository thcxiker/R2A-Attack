"""Attack success rate on the target router (Sec. 4.1).

ASR(s) = fraction of evaluation queries that the target routes to a model in
M_strong when suffix s is appended. ``clean`` (s = empty) is the no-attack
rate. Only the target's routing decisions are used (``BlackBoxTarget``).
"""

import logging
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm

from r2a.blackbox import BlackBoxTarget
from r2a.classifier import ModelClassifier
from r2a.utils import extract_text, query_source

logger = logging.getLogger(__name__)


def evaluate_suffixes(
    target: BlackBoxTarget,
    classifier: ModelClassifier,
    datasets: Dict[str, List[Dict]],
    suffixes: Dict[str, str],
    prefix_mode: bool = False,
) -> Dict:
    """Evaluate every suffix on every dataset; ``suffixes`` maps a label to a suffix string."""
    suffixes = {"clean": "", **{k: v for k, v in suffixes.items() if k != "clean"}}
    report = {}
    for ds_name, samples in datasets.items():
        per_suffix = {}
        for label, suffix in suffixes.items():
            hits, by_source, records = 0, defaultdict(lambda: [0, 0]), []
            for s in tqdm(samples, desc=f"{ds_name}/{label}", leave=False, disable=None):
                q, src = extract_text(s.get("question", "")), query_source(s)
                prompt, sfx = (suffix, q) if prefix_mode else (q, suffix)
                decision = target.decide(prompt, sfx, type=src)
                strong = classifier.classify(decision) == "strong"
                hits += strong
                by_source[src][0] += strong
                by_source[src][1] += 1
                records.append({
                    "question_id": s.get("question_id", s.get("index")),
                    "source": src,
                    "decision": decision,
                    "strong": strong,
                })
            per_suffix[label] = {
                "suffix": suffix,
                "asr": hits / max(1, len(samples)),
                "n": len(samples),
                "asr_by_source": {k: v[0] / v[1] for k, v in by_source.items()},
                "records": records,
            }
            logger.info("[%s] %-10s ASR = %.3f (%d/%d)", ds_name, label, hits / max(1, len(samples)), hits, len(samples))
        report[ds_name] = per_suffix
    return report


def summary_table(report: Dict) -> str:
    labels = sorted({l for ds in report.values() for l in ds}, key=lambda x: (x != "clean", x))
    lines = ["| dataset | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for ds_name, per_suffix in report.items():
        cells = [f"{per_suffix[l]['asr']:.3f}" if l in per_suffix else "-" for l in labels]
        lines.append(f"| {ds_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
