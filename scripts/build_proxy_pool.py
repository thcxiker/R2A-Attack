"""Build ``data/proxy_pool.json``: the prompts D_proxy used to query the target
router during surrogate training.

The pool mixes MMLU(-Pro), GSM8K and MT-Bench-101 prompts (the in-distribution
task families) and removes every prompt that appears in any evaluation set, so
D_proxy, D_suffix and D_eval are disjoint.

Inputs are JSON / JSONL files with one query per record: MMLU and GSM8K records
carry a ``question`` field, MT-Bench-101 records carry a ``history`` list of
``{"user", "bot"}`` turns (joined into one prompt).

    python scripts/build_proxy_pool.py \
        --mmlu mmlu.json --gsm8k gsm8k.json --mtbench mtbench101.jsonl \
        --per-source 2000 --out data/proxy_pool.json
"""

import argparse
import json
import random
from pathlib import Path


def read_records(path):
    with open(path, "r", encoding="utf-8") as f:
        if str(path).endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def prompt_of(record):
    if isinstance(record.get("question"), str):
        return record["question"]
    history = record.get("history")
    if isinstance(history, list):
        return " ".join(
            f"{t.get('user', '')} {t.get('bot', '')}" if isinstance(t, dict) else str(t) for t in history
        ).strip()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mmlu", required=True)
    ap.add_argument("--gsm8k", required=True)
    ap.add_argument("--mtbench", required=True)
    ap.add_argument("--per-source", type=int, default=2000)
    ap.add_argument("--exclude", nargs="*", default=[
        "data/in_distribution.json", "data/in_distribution_2.json", "data/in_distribution_3.json", "data/ood.json",
    ])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/proxy_pool.json")
    args = ap.parse_args()

    excluded = set()
    for path in args.exclude:
        for item in read_records(path):
            q = item["question"]
            excluded.add((" ".join(q) if isinstance(q, list) else str(q)).strip())

    rng = random.Random(args.seed)
    pool = []
    for source, path in (("mmlu", args.mmlu), ("gsm8k", args.gsm8k), ("mtbench", args.mtbench)):
        prompts, seen = [], set()
        for rec in read_records(path):
            p = prompt_of(rec)
            if p and p.strip() and p.strip() not in excluded and p not in seen:
                seen.add(p)
                prompts.append(p)
        rng.shuffle(prompts)
        kept = prompts[: args.per_source]
        pool += [{"question_id": f"{source}_{i}", "question": p, "source": source} for i, p in enumerate(kept)]
        print(f"{source}: {len(prompts)} candidate prompts, kept {len(kept)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(pool)} prompts to {args.out}")


if __name__ == "__main__":
    main()
