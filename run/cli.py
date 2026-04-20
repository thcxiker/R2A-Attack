import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from reroute.core.pair import RouterEncoderPair
from reroute.core.router import create_router
from reroute.core.encoder import EncoderFactory
from reroute.attacks.gcg import GCGTrainerConfig, RouterGCG
from reroute.core.ensemble import (
    LowRankSemanticAdapter,
    load_adapter_from_checkpoint,
)


def _expand_path(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def _load_yaml(path: str) -> Dict[str, Any]:
    config_path = Path(_expand_path(path))
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_router(router_config: Dict[str, Any]):
    router_type = router_config["router_type"]
    args = router_config.get("args", {})
    expanded_args = {
        key: _expand_path(value) if isinstance(value, str) else value
        for key, value in args.items()
    }
    return create_router(router_type, **expanded_args)


def _build_router_pairs(pairs_config: List[Dict[str, Any]]) -> List[RouterEncoderPair]:
    router_pairs = []
    for pair_config in pairs_config:
        router = _build_router(pair_config["router"])
        encoder = None
        encoder_name = pair_config.get("encoder_name")
        if encoder_name:
            encoder_device = pair_config.get("encoder_device")
            encoder = EncoderFactory.create_from_yaml(encoder_name, device=encoder_device)
        router_pairs.append(
            RouterEncoderPair(
                router=router,
                encoder=encoder,
                trainable=pair_config.get("trainable", True),
                name=pair_config.get("name", router.name),
            )
        )
    return router_pairs


def run_eval_from_config(config_path: str) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    eval_config = config["eval"]

    adapter_ckpt = _expand_path(eval_config["adapter_checkpoint"])
    adapter = LowRankSemanticAdapter()
    adapter, unified_names, target_names = load_adapter_from_checkpoint(
        adapter_ckpt,
        device=eval_config.get("adapter_device", "cuda"),
        text_embedder_name=eval_config.get(
            "adapter_text_embedder",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
    )

    router_pairs = _build_router_pairs(eval_config["router_encoder_pairs"])

    gcg_config = GCGTrainerConfig(**eval_config["gcg_config"])
    router_gcg = RouterGCG(
        router_encoder_pairs=router_pairs,
        config=gcg_config,
        adapter=adapter,
        unified_model_names=unified_names,
        target_model_names=target_names,
        loss_type=eval_config.get("loss_type", "hierarchical_distribution"),
        target_type=eval_config.get("target_type", "strong"),
    )

    target_router = _build_router(eval_config["target_router"])
    run_config = eval_config["gcg_runtime"]

    return train_eval_universal_suffix(
        router_gcg=router_gcg,
        target_router=target_router,
        test_set_path=_expand_path(run_config["test_set_path"]),
        output_report_path=_expand_path(run_config["output_report_path"]),
        train_ratio=run_config.get("train_ratio", 0.7),
        strong_abs_threshold=run_config.get("strong_abs_threshold", 0.2),
        strong_delta_threshold=run_config.get("strong_delta_threshold", 0.01),
        max_iterations_per_sample=run_config.get("max_iterations_per_sample", 10),
        max_stage_iterations=run_config.get("max_stage_iterations", 5),
        random_seed=run_config.get("random_seed", 42),
    )


def run_train_data_from_config(config_path: str) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    train_config = config["train"]

    dataset_config = train_config["unified_dataset"]
    train_data, test_data = create_unified_dataset(
        processed_files=[_expand_path(path) for path in dataset_config["processed_files"]],
        router_names=dataset_config["router_names"],
        candidate_lists=dataset_config["candidate_lists"],
        output_file=_expand_path(dataset_config["output_file"]),
        test_split=dataset_config.get("test_split", 0.2),
        random_seed=dataset_config.get("random_seed", 42),
    )

    return {"train": train_data, "test": test_data}




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reroute CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Evaluation / attack pipeline")
    eval_parser.add_argument("--config", required=True, help="Evaluation config file path")

    train_parser = subparsers.add_parser("train", help="Training / alignment data preparation")
    train_parser.add_argument("--config", required=True, help="Training config file path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "eval":
        run_eval_from_config(args.config)
    elif args.command == "train":
        run_train_data_from_config(args.config)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
