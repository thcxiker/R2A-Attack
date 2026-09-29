"""Black-box access to the target router (threat model of Sec. 2.2).

The attacker only observes *which model* the target router selects for a
query, and may issue at most Q queries while building the surrogate. All
access to the target in the black-box setting goes through
``BlackBoxTarget.decide``, which returns a model name and counts queries; the
router's scores never leave this class.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryBudgetExceeded(RuntimeError):
    pass


class BlackBoxTarget:
    def __init__(self, router, budget: Optional[int] = None):
        self._router = router
        self.budget = budget
        self.queries = 0

    @property
    def name(self) -> str:
        return self._router.name

    def candidate_pool(self) -> List[str]:
        """The candidate model names (public for commercial routers, see Table 1 of the paper)."""
        return self._router.get_model_list()

    def decide(self, prompt: str, suffix: str = "", type: Optional[str] = None) -> str:
        """Return the model the target routes ``prompt + suffix`` to."""
        if self.budget is not None and self.queries >= self.budget:
            raise QueryBudgetExceeded(f"query budget of {self.budget} target queries exhausted")
        self.queries += 1
        scores = self._router.route(prompt, suffix, type=type).detach().float().reshape(-1)
        return self._router.get_model_list()[int(scores.argmax())]


def _as_dict(value) -> Dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
    return {}


def load_decision_logs(paths: List[str]) -> List[Dict]:
    """Read logged routing decisions of a black-box router.

    Each file is a JSON list of records, or a dict with ``records`` (and
    optionally ``dataset_name``). A record needs the prompt (``prompt`` or
    ``origin_query``) and the selected model (``extra_fields.actual_model`` or
    ``model_name``), e.g. the OpenRouter logs used in the paper.
    """
    out = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            source = data.get("dataset_name") or data.get("dataset") or Path(path).stem
            records = data.get("records", [])
        else:
            source, records = Path(path).stem, data
        n = 0
        for r in records:
            prompt = r.get("prompt") or r.get("origin_query")
            model = _as_dict(r.get("extra_fields")).get("actual_model") or r.get("model_name")
            if prompt and model:
                out.append({"prompt": prompt, "source": source, "decision": model,
                            "origin_query": r.get("origin_query") or ""})
                n += 1
        logger.info("Loaded %d logged decisions from %s", n, path)
    return out
