"""
Look at the prompt, and at what collapsed.

few_shot accuracy falls monotonically with exemplar count - 45.8% at zero,
22.0% at k=5, 12.0% at k=20, 5.7% at k=77 - and free-form and constrained
decoding agree on 3,079 of 3,080 items, so it is not a decoding artefact.
At k=20, 37% of all predictions were a single label: `Refund_not_showing_up`,
which was exemplar #20 of 20. That is recency anchoring, from one draw of
exemplars.

One draw cannot distinguish "exemplars hurt" from "this draw hurt". Run the
seed variants and pass them all here:

    python src/inspect_prompts.py --configs few_shot_k20 few_shot_k20_s2 few_shot_k20_s3

The cross-config table at the end is the test. If each seed collapses onto
its own last exemplar, recency is the mechanism. If they collapse onto the
same label regardless of position, it is something about that label. If
accuracy swings widely, the finding is exemplar-dependent and weak.

Config definitions (k, seed) are read from src/04a_baselines.py rather than
duplicated here - a copy would drift from what actually ran, and then this
script would reconstruct the wrong exemplars and quietly mislead.
"""

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts as P  # noqa: E402

RUNS = Path("reports/runs")
SPLITS = Path("eval/splits")
SAMPLE_SEED = 20260821


def load_configs():
    """Read CONFIGS from 04a_baselines.py - the module name starts with a
    digit, so it cannot be imported normally."""
    path = Path(__file__).resolve().parent / "04a_baselines.py"
    spec = importlib.util.spec_from_file_location("_baselines", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIGS


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def show_prompt(text, head=16, tail=14):
    lines = text.split("\n")
    if len(lines) <= head + tail:
        print("    " + "\n    ".join(lines))
        return
    print("    " + "\n    ".join(lines[:head]))
    print(f"    ... [{len(lines) - head - tail} lines omitted] ...")
    print("    " + "\n    ".join(lines[-tail:]))


def exemplars_for(cfg, pool):
    return P.fixed_exemplars(pool, cfg["k"], seed=cfg.get("seed", 20260821))


def analyse(name, cfg, pool, labels, test, n_show, show_prompt_text):
    ex_labels = [e["label"] for e in exemplars_for(cfg, pool)]
    k = cfg["k"]
    path = RUNS / f"{name}__free_form" / "predictions.jsonl"
    if not path.exists():
        print(f"\n  {name}: no predictions at {path}")
        return None
    preds = read_jsonl(path)
    by_id = {r["id"]: r for r in preds}

    print(f"\n{'=' * 68}\n{name}   k={k}   seed={cfg.get('seed', 20260821)}"
          f"\n{'=' * 68}")
    print(f"  last exemplar shown: {ex_labels[-1]}")

    if show_prompt_text:
        msgs = P.few_shot(test[0]["text"], labels, exemplars_for(cfg, pool))
        print("\n  [system]\n    " + msgs[0]["content"])
        print("  [user]")
        show_prompt(msgs[1]["content"])

    # test.jsonl is class-ordered, so a prefix is one intent and would make
    # any repeated answer look like collapse. Sample across classes.
    shown = list(test)
    random.Random(SAMPLE_SEED).shuffle(shown)
    print(f"\n  {n_show} queries sampled across classes:")
    for r in shown[:n_show]:
        rec = by_id.get(r["id"])
        if not rec:
            continue
        pred = rec.get("pred")
        tag = ""
        if pred in ex_labels:
            tag = f"   [exemplar #{ex_labels.index(pred) + 1}/{k}]"
        print(f"    gold {r['label']:38s} -> {str(pred):38s}{tag}")

    vals = [r["pred"] for r in preds if r["pred"] is not None]
    counts = Counter(vals)
    top1, n1 = counts.most_common(1)[0]
    share2 = sum(n for _, n in counts.most_common(2)) / len(vals)
    acc = sum(1 for r in preds if r["verdict"] == "correct") / len(preds)

    print(f"\n  accuracy                    {acc * 100:5.1f}%")
    print(f"  distinct labels predicted   {len(counts):3d} / 77")
    print(f"  most-predicted label        {top1}  "
          f"({n1 / len(vals) * 100:.1f}% of all predictions)")
    print(f"  top-2 labels cover          {share2 * 100:5.1f}% of predictions")

    in_ex = [r for r in preds if r["pred"] in ex_labels]
    if in_ex:
        pos = Counter(ex_labels.index(r["pred"]) for r in in_ex)
        last = sum(v for p, v in pos.items() if p >= k * 0.9)
        print(f"  predictions in last 10% of exemplar list: "
              f"{last / len(in_ex) * 100:5.1f}%   (uniform = 10%)")
        print("\n  most-predicted exemplars, by position:")
        for p, v in pos.most_common(5):
            print(f"    #{p + 1:3d}/{k}  {ex_labels[p]:42s} {v:5,d}")

    return {
        "name": name, "k": k, "seed": cfg.get("seed", 20260821),
        "last_exemplar": ex_labels[-1], "top1": top1,
        "top1_share": n1 / len(vals), "top2_share": share2,
        "accuracy": acc, "distinct": len(counts),
        "top1_is_last": top1 == ex_labels[-1],
        "top1_pos": (ex_labels.index(top1) + 1) if top1 in ex_labels else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["few_shot_k20"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the full rendered prompt for the first config")
    args = ap.parse_args()

    CONFIGS = load_configs()
    labels, pool = P.load_labels(), P.load_pool()
    test = read_jsonl(SPLITS / "test.jsonl")

    rows = []
    for i, name in enumerate(args.configs):
        cfg = CONFIGS.get(name)
        if cfg is None or cfg.get("template") != "few_shot":
            print(f"\n  {name}: not a few_shot config in 03_baselines.CONFIGS")
            continue
        r = analyse(name, cfg, pool, labels, test, args.n,
                    args.show_prompt and i == 0)
        if r:
            rows.append(r)

    if len(rows) < 2:
        return

    print(f"\n{'=' * 68}\ncross-config: does the collapse follow the last "
          f"exemplar?\n{'=' * 68}")
    print(f"  {'config':18s} {'seed':>9s} {'acc':>6s} {'top-1 prediction':30s}"
          f" {'pos':>5s} {'share':>6s}  last?")
    for r in rows:
        pos = f"{r['top1_pos']}/{r['k']}" if r["top1_pos"] else "-"
        print(f"  {r['name']:18s} {r['seed']:9d} {r['accuracy'] * 100:5.1f}% "
              f"{r['top1']:30s} {pos:>5s} {r['top1_share'] * 100:5.1f}%  "
              f"{'YES' if r['top1_is_last'] else 'no'}")

    n_last = sum(r["top1_is_last"] for r in rows)
    same = len({r["top1"] for r in rows}) == 1
    accs = [r["accuracy"] for r in rows]
    print(f"\n  collapsed onto the last exemplar: {n_last} / {len(rows)}")
    print(f"  same label across all seeds:       {same}")
    print(f"  accuracy spread across seeds:      "
          f"{(max(accs) - min(accs)) * 100:.1f} pp")
    print("\n  All on the last exemplar -> recency anchoring, mechanism found.")
    print("  Same label regardless of position -> a property of that label.")
    print("  Wide accuracy spread -> the result is draw-dependent and weak.")


if __name__ == "__main__":
    main()
