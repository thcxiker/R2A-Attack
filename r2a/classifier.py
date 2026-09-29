"""Strong / weak model partition (M_strong vs M_weak).

The partition is read from ``configs/model_classification.yaml``: explicit
``strong_models`` / ``weak_models`` lists are checked first, then regex rules,
and anything left falls back to ``default_classification``.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch
import yaml


class ModelClassifier:
    def __init__(self, config_path: Optional[str] = None, default_classification: str = "weak"):
        self.strong_models, self.weak_models = set(), set()
        self.strong_patterns, self.weak_patterns = [], []
        self.default_classification = default_classification
        if config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            self.strong_models = set(config.get("strong_models") or [])
            self.weak_models = set(config.get("weak_models") or [])
            rules = config.get("classification_rules") or {}
            self.strong_patterns = [re.compile(p) for p in rules.get("strong_patterns") or []]
            self.weak_patterns = [re.compile(p) for p in rules.get("weak_patterns") or []]
            self.default_classification = config.get("default_classification", default_classification)

    def classify(self, model_name: str) -> str:
        if model_name in self.strong_models:
            return "strong"
        if model_name in self.weak_models:
            return "weak"
        for pattern in self.strong_patterns:
            if pattern.match(model_name):
                return "strong"
        for pattern in self.weak_patterns:
            if pattern.match(model_name):
                return "weak"
        return self.default_classification

    def classify_list(self, model_list: List[str]) -> Tuple[List[str], List[str]]:
        strong, weak = [], []
        for model in model_list:
            (strong if self.classify(model) == "strong" else weak).append(model)
        return strong, weak

    def strong_weak_indices(self, model_list: List[str]) -> Tuple[List[int], List[int]]:
        strong = [i for i, m in enumerate(model_list) if self.classify(m) == "strong"]
        weak = [i for i, m in enumerate(model_list) if self.classify(m) != "strong"]
        return strong, weak


def routing_outcome(logits: torch.Tensor, model_list: List[str], classifier: ModelClassifier) -> Dict:
    """Summarize a router's decision on one query.

    A query counts as routed to a strong model when the highest-scoring strong
    model scores at least as high as the highest-scoring weak model, i.e. the
    router's argmax lies in M_strong (ties count as strong).
    """
    logits = logits.detach().float().reshape(-1)
    if logits.numel() != len(model_list):
        raise ValueError(f"router returned {logits.numel()} scores for {len(model_list)} models")
    probs = torch.softmax(logits, dim=0)
    strong_idx, weak_idx = classifier.strong_weak_indices(model_list)
    strong = bool(strong_idx) and bool(weak_idx) and (
        logits[strong_idx].max().item() >= logits[weak_idx].max().item()
    )
    return {
        "strong": bool(strong),
        "strong_prob": probs[strong_idx].sum().item() if strong_idx else 0.0,
        "weak_prob": probs[weak_idx].sum().item() if weak_idx else 0.0,
        "rank1": model_list[int(probs.argmax().item())],
    }
