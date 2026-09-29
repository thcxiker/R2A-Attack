"""End-to-end R2A pipeline: surrogate training -> suffix optimization -> evaluation."""

import gc
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

from r2a.attack.gcg import EnsembleGCG, GCGConfig, Member
from r2a.attack.optimize import StageConfig, optimize_universal_suffix
from r2a.classifier import ModelClassifier
from r2a.config import resolve_path, validate_config
from r2a.evaluate import evaluate_suffixes, summary_table
from r2a.routers import ROUTER_TYPES, create_router
from r2a.routers.lowrank import LowRankRouter
from r2a.surrogate.adapter import load_adapter_checkpoint, save_adapter_checkpoint
from r2a.blackbox import BlackBoxTarget
from r2a.surrogate.train import SurrogateConfig, collect_decisions, train_surrogate
from r2a.utils import load_json, save_json, set_seed, setup_logging, split_suffix_set

logger = logging.getLogger(__name__)


def _dataclass_kwargs(cls, section: Dict) -> Dict:
    fields = set(inspect.signature(cls).parameters)
    unknown = set(section) - fields
    if unknown:
        raise ValueError(f"unknown {cls.__name__} options: {sorted(unknown)}")
    return dict(section)


class Pipeline:
    def __init__(self, cfg: Dict):
        errors = validate_config(cfg)
        if errors:
            raise ValueError("invalid config:\n  - " + "\n  - ".join(errors))
        self.cfg = cfg
        exp = cfg.get("experiment", {})
        self.seed = exp.get("seed", 42)
        self.device = exp.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.out = Path(resolve_path(exp.get("output_dir") or f"outputs/{exp.get('name', 'r2a')}"))
        self.out.mkdir(parents=True, exist_ok=True)
        setup_logging(exp.get("log_level", "INFO"), str(self.out / "run.log"))
        with open(self.out / "config.resolved.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        set_seed(self.seed)

        self.classifier = ModelClassifier(resolve_path(cfg.get("model_classification", "configs/model_classification.yaml")))
        self._routers: Dict[str, object] = {}
        self.surrogate_path = self.out / "surrogate.pt"

    # ------------------------------------------------------------------ routers
    def router(self, key: str, consider_cost: bool = False):
        cache_key = (key, consider_cost)
        if cache_key in self._routers:
            return self._routers[cache_key]
        spec = self.cfg["routers"][key]
        rtype = spec["type"]
        module_name, class_name = ROUTER_TYPES[rtype]
        import importlib

        cls = getattr(importlib.import_module(module_name), class_name)
        params = inspect.signature(cls).parameters
        kwargs = dict(spec.get("args", {}))
        kwargs["name"] = key
        # Paths relative to the repo root are resolved; anything else (e.g. a
        # Hugging Face repo id) is passed through unchanged.
        for k in ("checkpoint_path", "model_path", "thresholds_file", "cost_file", "context_path"):
            if isinstance(kwargs.get(k), str) and Path(resolve_path(kwargs[k])).exists():
                kwargs[k] = resolve_path(kwargs[k])
        if "consider_cost" in params:
            kwargs["consider_cost"] = consider_cost
        logger.info("Loading router %s (%s, consider_cost=%s)", key, rtype, consider_cost)
        router = create_router(rtype, **kwargs)
        self._routers[cache_key] = router
        return router

    def target(self):
        t = self.cfg["target"]
        return self.router(t["router"], consider_cost=t.get("consider_cost", True))

    def member_cost(self, key: str) -> bool:
        """Whether a member uses its cost-aware scores (see `member_consider_cost` in base.yaml)."""
        return bool(self.cfg["routers"][key].get("member_consider_cost", False))

    def members(self) -> List:
        return [self.router(k, consider_cost=self.member_cost(k)) for k in self.cfg["ensemble"]["members"]]

    def release(self, key: str, consider_cost: bool) -> None:
        self._routers.pop((key, consider_cost), None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ data
    def suffix_split(self):
        data = self.cfg["data"]
        items = load_json(resolve_path(data["suffix_set"]))
        return split_suffix_set(items, data.get("train_ratio", 0.7), self.seed)

    def eval_sets(self, max_samples: Optional[int] = None) -> Dict[str, List[Dict]]:
        _, d_eval = self.suffix_split()
        sets = {"in_distribution": d_eval}
        for name, spec in (self.cfg["data"].get("eval_sets") or {}).items():
            sets[name] = load_json(resolve_path(spec["path"]))
        if max_samples:
            sets = {k: v[:max_samples] for k, v in sets.items()}
        return sets

    def held_out_prompts(self) -> set:
        """Texts of D_suffix and every evaluation set; they are never used to train the surrogate."""
        from r2a.utils import extract_text

        d_suffix, _ = self.suffix_split()
        texts = {extract_text(s["question"]).strip() for s in d_suffix}
        for samples in self.eval_sets().values():
            texts |= {extract_text(s["question"]).strip() for s in samples}
        return texts

    # ------------------------------------------------------------------ stages
    def train_surrogate(self) -> Dict:
        scfg = SurrogateConfig(**_dataclass_kwargs(SurrogateConfig, self.cfg.get("surrogate", {})))
        if scfg.decision_logs:
            scfg.decision_logs = [resolve_path(p) for p in scfg.decision_logs]
        pool = load_json(resolve_path(self.cfg["data"]["proxy_pool"]))
        target = None
        if not scfg.decision_logs or scfg.target_pool == "declared":
            # without logs: at most Q decisions; with logs + declared pool: only the candidate list is read
            budget = 0 if scfg.decision_logs else scfg.query_budget
            target = BlackBoxTarget(self.target(), budget=budget)
        samples, target_models, n_queries = collect_decisions(
            target, pool, scfg, self.seed, exclude=self.held_out_prompts()
        )
        strong, weak = self.classifier.classify_list(target_models)
        logger.info(
            "Target pool for the surrogate: %d models (%d strong, %d weak); %d target queries; %d labeled prompts",
            len(target_models), len(strong), len(weak), n_queries, len(samples),
        )
        if not strong or not weak:
            raise ValueError(
                "the surrogate's target pool needs both strong and weak models; observed only "
                f"{'strong' if strong else 'weak'} decisions. Use more queries or surrogate.target_pool=declared."
            )

        members = self.members()
        unified = sorted({m for r in members for m in r.get_model_list()})
        adapter, summary = train_surrogate(members, samples, unified, target_models, scfg, self.device, self.seed)
        summary["target_queries"] = n_queries
        summary["query_budget"] = scfg.query_budget
        save_adapter_checkpoint(
            adapter, self.surrogate_path, unified, target_models,
            router_names=list(self.cfg["ensemble"]["members"]),
            text_embedder_name=scfg.text_embedder, history=summary,
        )
        save_json(summary, str(self.out / "surrogate_summary.json"))
        logger.info(
            "Surrogate: best epoch %s | val top-1 agreement %s | router weights %s",
            summary["best_epoch"], summary["val_top1_agreement"], summary["router_weights"],
        )
        return summary

    def build_attack(self, surrogate_path: Optional[str] = None) -> EnsembleGCG:
        path = surrogate_path or self.surrogate_path
        adapter, unified, target_names, router_names = load_adapter_checkpoint(path, self.device)
        members_cfg = list(self.cfg["ensemble"]["members"])
        if router_names and list(router_names) != members_cfg:
            raise ValueError(f"surrogate was trained with members {router_names}, config lists {members_cfg}")
        acfg = dict(self.cfg.get("attack", {}))
        gcg_cfg = GCGConfig(**{k: acfg[k] for k in inspect.signature(GCGConfig).parameters if k in acfg})
        members = []
        for key in members_cfg:
            r = self.router(key, consider_cost=self.member_cost(key))
            members.append(Member(key, r, r.get_internal_encoder()))
        lowrank = LowRankRouter(adapter, target_names, device=self.device)
        return EnsembleGCG(
            members=members,
            lowrank=Member("lowrank", lowrank, lowrank.get_internal_encoder()),
            adapter=adapter,
            unified_model_names=unified,
            target_model_names=target_names,
            classifier=self.classifier,
            config=gcg_cfg,
            loss=acfg.get("loss", "strong_promotion"),
            device=self.device,
        )

    def optimize(self, surrogate_path: Optional[str] = None, resume: bool = False) -> Dict:
        gcg = self.build_attack(surrogate_path)
        acfg = self.cfg.get("attack", {})
        stage_cfg = StageConfig(**{k: acfg[k] for k in inspect.signature(StageConfig).parameters if k in acfg})
        d_suffix, _ = self.suffix_split()
        set_seed(self.seed)
        return optimize_universal_suffix(gcg, d_suffix, stage_cfg, str(self.out), resume=resume)

    def evaluate(self, extra_suffixes: Optional[Dict[str, str]] = None) -> Dict:
        ecfg = self.cfg.get("evaluation", {})
        suffixes = dict(ecfg.get("suffixes") or {})
        trained = self.out / "trained_suffix.json"
        if trained.exists():
            suffixes["r2a"] = load_json(str(trained))["suffix"]
        suffixes.update(extra_suffixes or {})
        if not self.cfg["target"].get("router"):
            raise ValueError("evaluation needs a target router (target.router is empty)")
        report = evaluate_suffixes(
            BlackBoxTarget(self.target()), self.classifier, self.eval_sets(ecfg.get("max_samples")), suffixes,
            prefix_mode=self.cfg.get("attack", {}).get("prefix_mode", False),
        )
        save_json(report, str(self.out / "eval_report.json"))
        table = summary_table(report)
        (self.out / "eval_summary.md").write_text(table + "\n", encoding="utf-8")
        logger.info("Attack success rate on target '%s':\n%s", self.cfg["target"]["router"], table)
        return report

    def run(self) -> None:
        self.train_surrogate()
        self.optimize()
        self.evaluate()
