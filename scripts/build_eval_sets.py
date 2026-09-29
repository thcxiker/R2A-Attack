"""Build the in-distribution query sets ``data/in_distribution*.json``.

Queries are drawn *uniformly at random* from MMLU(-Pro), GSM8K and MT-Bench;
no router is consulted, so the sets carry no information about any router.
Each set is later split 70/30 into D_suffix and D_eval (seed 42).

    python scripts/build_eval_sets.py --mmlu mmlu.json --gsm8k gsm8k.json --mtbench mt_bench.json

MMLU / GSM8K records need a ``question`` field, MT-Bench records ``turns``.
"""

import argparse
import ast
import json
import random

# (seed, #MMLU, #GSM8K, #MT-Bench) of the three sets; same sizes and source mix as in the paper
SETS = {
    "data/in_distribution.json": (1, 285, 215, 80),
    "data/in_distribution_2.json": (2, 339, 211, 50),
    "data/in_distribution_3.json": (3, 337, 212, 51),
}


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mmlu", required=True)
    ap.add_argument("--gsm8k", required=True)
    ap.add_argument("--mtbench", required=True)
    args = ap.parse_args()

    pools = {"mmlu": [], "gsm8k": [], "mtbench": []}
    seen = set()
    for i, r in enumerate(load(args.mmlu)):
        q = r.get("question")
        if isinstance(q, str) and q.strip() and q not in seen:
            seen.add(q)
            pools["mmlu"].append({"question_id": r.get("question_id", i), "question": q, "category": r.get("category")})
    for i, r in enumerate(load(args.gsm8k)):
        q = r.get("question")
        if isinstance(q, str) and q.strip() and q not in seen:
            seen.add(q)
            pools["gsm8k"].append({"question_id": f"auto_{i}", "question": q, "category": "math"})
    for r in load(args.mtbench):
        turns = r["turns"]
        turns = ast.literal_eval(turns) if isinstance(turns, str) else turns
        pools["mtbench"].append({"question_id": r["question_id"], "question": list(turns), "category": r.get("category")})
    print({k: len(v) for k, v in pools.items()})

    for out, (seed, n_mmlu, n_gsm, n_mt) in SETS.items():
        rng = random.Random(seed)
        items = []
        for source, n in (("mmlu", n_mmlu), ("gsm8k", n_gsm), ("mtbench", n_mt)):
            items += [dict(x, source=source, selection_type="random") for x in rng.sample(pools[source], n)]
        for i, x in enumerate(items):
            x["test_idx"] = i
        with open(out, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"wrote {len(items)} queries to {out}")


if __name__ == "__main__":
    main()
