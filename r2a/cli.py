"""Command-line interface.

    python -m r2a run       --config configs/paper/routellm_bert.yaml
    python -m r2a surrogate --config ...    # stage 1: query target, train surrogate
    python -m r2a attack    --config ...    # stage 2: optimize the universal suffix
    python -m r2a evaluate  --config ...    # stage 3: ASR on the target router
    python -m r2a check     --config ...    # validate a config without loading models

Any config value can be overridden with ``--set section.key=value``.
"""

import argparse
import sys

from r2a.config import load_config, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="r2a", description="Route-to-Rome Attack (R2A)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="YAML experiment config")
        p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config value, e.g. --set attack.max_total_steps=100")
        return p

    add("run", "surrogate training + suffix optimization + evaluation")
    add("surrogate", "query the target and train the hybrid ensemble surrogate")
    p = add("attack", "optimize a universal suffix against the surrogate")
    p.add_argument("--surrogate", default=None, help="surrogate checkpoint (default: <output_dir>/surrogate.pt)")
    p.add_argument("--resume", action="store_true", help="resume from <output_dir>/suffix_progress.json")
    p = add("evaluate", "measure attack success rate on the target router")
    p.add_argument("--suffix", action="append", default=[], metavar="LABEL=SUFFIX",
                   help="extra suffix to evaluate, e.g. --suffix 'mine=hello world'")
    add("check", "validate the config (no models are loaded)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    if args.command == "check":
        errors = validate_config(cfg)
        for e in errors:
            print(f"[error] {e}")
        if not errors:
            print(f"OK: target={cfg['target']['router']} members={cfg['ensemble']['members']}")
        return 1 if errors else 0

    from r2a.pipeline import Pipeline

    pipe = Pipeline(cfg)
    if args.command == "run":
        pipe.run()
    elif args.command == "surrogate":
        pipe.train_surrogate()
    elif args.command == "attack":
        pipe.optimize(surrogate_path=args.surrogate, resume=args.resume)
    elif args.command == "evaluate":
        extra = {}
        for item in args.suffix:
            label, sep, value = item.partition("=")
            extra[label if sep else f"suffix{len(extra)}"] = value if sep else item
        pipe.evaluate(extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
