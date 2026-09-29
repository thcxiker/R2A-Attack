"""YAML configs with single inheritance (``base: other.yaml``) and CLI overrides."""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _deep_merge(load_config(str((path.parent / base).resolve())), cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key.sub=value, got '{item}'")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def resolve_path(value: Optional[str]) -> Optional[str]:
    """Relative paths in configs are relative to the repository root."""
    if value is None:
        return None
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else REPO_ROOT / p)


def validate_config(cfg: Dict) -> List[str]:
    """Cheap consistency checks that do not load any model."""
    errors = []
    routers = cfg.get("routers", {})
    target = cfg.get("target", {}).get("router")
    members = cfg.get("ensemble", {}).get("members", [])
    decision_logs = cfg.get("surrogate", {}).get("decision_logs")
    if target is None and not decision_logs:
        errors.append("target.router is empty; that is only allowed with surrogate.decision_logs")
    elif target is not None and target not in routers:
        errors.append(f"target router '{target}' is not defined under routers:")
    for m in members:
        if m not in routers:
            errors.append(f"ensemble member '{m}' is not defined under routers:")
    if target in members:
        errors.append(f"target '{target}' is also an ensemble member (black-box: the target must not be in the surrogate)")
    for m in members:
        if routers.get(m, {}).get("type") == "routellm" and routers[m].get("args", {}).get("type") in ("mf", "sw_ranking"):
            errors.append(f"member '{m}' uses API embeddings and cannot provide gradients")
    data = cfg.get("data", {})
    for key in ("proxy_pool", "suffix_set"):
        p = resolve_path(data.get(key))
        if not p or not Path(p).exists():
            errors.append(f"data.{key} not found: {p}")
    for name, spec in (data.get("eval_sets") or {}).items():
        p = resolve_path(spec.get("path")) if isinstance(spec, dict) else None
        if p and not Path(p).exists():
            errors.append(f"data.eval_sets.{name} not found: {p}")
    for p in decision_logs or []:
        if not Path(resolve_path(p)).exists():
            errors.append(f"surrogate.decision_logs file not found: {resolve_path(p)}")
    return errors
