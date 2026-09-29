"""Small shared helpers: seeding, text extraction, JSON IO, logging."""

import json
import logging
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("r2a")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_text(question: Any) -> str:
    """Queries are either a string or a list of turns (MT-Bench); turns are joined."""
    if isinstance(question, list):
        return " ".join(str(x) for x in question)
    return str(question)


def query_source(sample: dict) -> Optional[str]:
    """Dataset name of a query; GraphRouter uses it to pick a task description."""
    return sample.get("source") or sample.get("type")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def split_suffix_set(items: List[dict], train_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    """Split the suffix-optimization set into D_suffix / D_eval.

    Deterministic for a given seed (``random.seed(seed); random.shuffle``).
    """
    rng_state = random.getstate()
    random.seed(seed)
    shuffled = items[:]
    random.shuffle(shuffled)
    random.setstate(rng_state)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]
