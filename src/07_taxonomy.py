"""
Stage 7 - failure taxonomy.

Stage 1 made a written prediction before any model existed:

    The largest failure cluster will be the intent pairs already visible
    in the probe - verify_my_identity/why_verify_identity,
    card_arrival/card_delivery_estimate, exchange_charge/exchange_rate -
    and they will persist at every rung, because more examples of a
    genuinely ambiguous boundary do not resolve it.

Every disagreement sample printed during stages 3 to 6 has been one of
those pairs. This counts them instead of eyeballing them.

Read on DEV. Test is scored once per frozen configuration and never
inspected - looking at test failures to characterise them is how a test
set stops being one.

What it reports:

  confusion pairs     which (gold, predicted) pairs account for the errors,
                      and how concentrated the errors are
  persistence         whether the same pairs dominate at 154 examples and
                      at 9,387, or whether the profile changes with data
  symmetry            whether A->B errors are matched by B->A. A symmetric
                      pair is an ambiguous boundary; a one-way pair is a
                      model preferring one label, which is a different
                      problem with a different fix
  gold agreement      for the top pairs, whether the near-duplicate
                      evidence from stage 1 shows the training data itself
                      disagrees about the boundary

    python src/07_taxonomy.py
    python src/07_taxonomy.py --configs qwen3lora_rungfull_ep8_seed1
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SPLITS = Path("eval/splits")
RUNS = Path("reports/runs")

# Pairs named in stage 1, before any model was run.
PREDICTED = [
    ("verify_my_identity", "why_verify_identity"),
    ("card_arrival", "card_delivery_estimate"),
    ("exchange_charge", "exchange_rate"),
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def norm_pair(a, b):
    return tuple(sorted((a, b)))


def load_predictions(runs, config, regime="constrained"):
    p = runs / f"{config}__{regime}" / "predictions.jsonl"
    if not p.exists():
        return None
    return read_jsonl(p)


def taxonomy(rows):
    """Errors grouped by unordered confusion pair."""
    errs = [r for r in rows if r["verdict"] != "correct"]
    pairs = Counter()
    directed = Counter()
    for r in errs:
        pred = r.get("pred") or "<unparseable>"
        pairs[norm_pair(r["gold"], pred)] += 1
        directed[(r["gold"], pred)] += 1
    return errs, pairs, directed


def report_config(name, rows, top=12):
    errs, pairs, directed = taxonomy(rows)
    n, n_err = len(rows), len(errs)
    print(f"\n  {name}   {n_err:,d} errors of {n:,d} "
          f"({n_err / n * 100:.1f}%)")
    if not n_err:
        return pairs

    cum = 0
    print(f"    {'n':>4s}  {'share':>6s} {'cum':>6s}  {'A -> B':<34s} "
          f"{'B -> A':>7s}  pair")
    for pair, count in pairs.most_common(top):
        a, b = pair
        ab, ba = directed[(a, b)], directed[(b, a)]
        cum += count
        tag = ""
        if norm_pair(*pair) in {norm_pair(*p) for p in PREDICTED}:
            tag = "  <- predicted in stage 1"
        sym = "symmetric" if min(ab, ba) > 0 else "one-way"
        print(f"    {count:4d}  {count / n_err * 100:5.1f}% "
              f"{cum / n_err * 100:5.1f}%  {a:<34s} {ba:>7d}  {b}"
              f"   [{sym}]{tag}")

    top10 = sum(c for _, c in pairs.most_common(10))
    print(f"    top 10 pairs cover {top10 / n_err * 100:.1f}% of errors; "
          f"{len(pairs)} distinct pairs over 77 labels "
          f"({77 * 76 // 2} possible)")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--regime", default="constrained")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    dev = read_jsonl(SPLITS / "dev.jsonl")
    dev_ids = {r["id"] for r in dev}
    near = {r["id"]: r for r in read_jsonl(SPLITS / "near_dup_dev.jsonl")}

    # Default: the LoRA scaling curve plus the encoder at full data, so
    # persistence can be read across data sizes and across model families.
    configs = args.configs or [
        "qwen3lora_rung02_ep8_seed1",
        "qwen3lora_rung08_ep8_seed1",
        "qwen3lora_rungfull_ep8_seed1",
        "robertabase_rungfull_seed1",
    ]

    print("Failure taxonomy, scored on DEV predictions where available.")
    print("Test predictions are used only if a config has no dev run, and")
    print("that is flagged - test is not a set you characterise failures on.")

    all_pairs = {}
    for cfg in configs:
        rows = load_predictions(runs, cfg, args.regime)
        if rows is None:
            print(f"\n  {cfg}: no predictions")
            continue
        on_dev = [r for r in rows if r["id"] in dev_ids]
        if on_dev:
            source, use = "dev", on_dev
        else:
            source, use = "TEST (no dev run available)", rows
        print(f"\n{'=' * 68}\n{cfg}  [{source}]\n{'=' * 68}")
        all_pairs[cfg] = report_config(cfg, use, args.top)

    # ------------------------------------------------------ persistence
    if len(all_pairs) > 1:
        print(f"\n{'=' * 68}\npersistence: do the same pairs dominate at "
              f"every data size?\n{'=' * 68}")
        union = Counter()
        for pairs in all_pairs.values():
            for pair, c in pairs.most_common(10):
                union[pair] += 1
        print(f"  {'pair':<58s} " +
              " ".join(f"{c.split('_')[1][:6]:>7s}" for c in all_pairs))
        for pair, _ in union.most_common(15):
            cells = []
            for cfg, pairs in all_pairs.items():
                total = sum(pairs.values()) or 1
                cells.append(f"{pairs.get(pair, 0) / total * 100:6.1f}%")
            mark = " <-" if pair in {norm_pair(*p) for p in PREDICTED} else ""
            print(f"  {pair[0][:28]:<28s} {pair[1][:28]:<28s} " +
                  " ".join(cells) + mark)
        print("\n  Cells are each pair's share of that config's errors.")
        print("  <- marks a pair named in stage 1 before any model existed.")

    # -------------------------------------------------- stage 1 scoring
    print(f"\n{'=' * 68}\nstage 1 prediction, scored\n{'=' * 68}")
    for cfg, pairs in all_pairs.items():
        total = sum(pairs.values()) or 1
        ranked = [p for p, _ in pairs.most_common()]
        hits = []
        for a, b in PREDICTED:
            pair = norm_pair(a, b)
            if pair in pairs:
                hits.append((a, b, pairs[pair],
                             ranked.index(pair) + 1,
                             pairs[pair] / total * 100))
        share = sum(h[2] for h in hits) / total * 100
        print(f"\n  {cfg}")
        if not hits:
            print("    none of the three predicted pairs appear")
            continue
        for a, b, c, rank, pct in hits:
            print(f"    {a} / {b}: {c} errors, rank {rank}, {pct:.1f}%")
        print(f"    combined: {share:.1f}% of this config's errors")

    # ------------------------------------------------ data disagreement
    print(f"\n{'=' * 68}\ndoes the training data itself disagree?\n{'=' * 68}")
    diff = [r for r in near.values() if r["twin_sim"] >= 0.85
            and r["twin_label"] != next(
                (d["label"] for d in dev if d["id"] == r["id"]), None)]
    print(f"  dev items whose nearest training twin (sim >= 0.85) carries a "
          f"different label: {len(diff)}")
    pairs = Counter()
    label_of = {d["id"]: d["label"] for d in dev}
    for r in diff:
        pairs[norm_pair(label_of[r["id"]], r["twin_label"])] += 1
    for pair, c in pairs.most_common(8):
        mark = "  <- predicted" if pair in {norm_pair(*p) for p in PREDICTED} \
            else ""
        print(f"    {c:3d}  {pair[0]:<34s} {pair[1]}{mark}")
    print("\n  These are boundaries the labelled data does not draw")
    print("  consistently. No model can be expected to draw them either.")


if __name__ == "__main__":
    main()
