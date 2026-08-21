"""
Paired comparison between two runs.

Every headline in this project is a difference between two configurations
scored on the same 3,080 items. That is a paired design, so the unpaired
error bar on each accuracy is the wrong instrument - what matters is the
items the two runs disagree on, and there are usually far fewer of those
than the accuracies alone suggest.

Reports, for the full test set and the near-duplicate-free subset:

  delta          B accuracy minus A accuracy
  discordant     items the two runs answer differently (b + c)
  McNemar p      probability of a split this lopsided by chance
  MDD            smallest difference this many discordant pairs gives
                 80% power to find - the quantity stage 0 predicted before
                 any of this existed

MDD frames NULL results: "no difference found, and we could have found one
of at least X". It must NOT be used to veto a result that reached
significance - that is post-hoc power analysis, and it is a mistake. If p
is small and the CI excludes zero, the difference is demonstrated whether
or not the study was well powered to find it.

    python src/compare.py reports/runs/zero_shot__free_form \\
                          reports/runs/zero_shot__constrained
"""

import argparse
import json
import math
from pathlib import Path

SPLITS = Path("eval/splits")
Z_ALPHA = 1.959963985
Z_BETA = 0.8416212336


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_run(d):
    rows = read_jsonl(Path(d) / "predictions.jsonl")
    return {r["id"]: r for r in rows}


def clean_ids():
    manifest = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    thr = manifest["near_dup_threshold_recorded"]
    return {r["id"] for r in read_jsonl(SPLITS / "near_dup_test.jsonl")
            if r["twin_sim"] < thr}


def mcnemar_p(b, c):
    """Exact binomial two-sided test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(a, b, subset=None):
    ids = sorted(set(a) & set(b))
    if subset is not None:
        ids = [i for i in ids if i in subset]
    n = len(ids)
    if n == 0:
        raise ValueError("no overlapping ids")

    a_ok = {i: a[i]["verdict"] == "correct" for i in ids}
    b_ok = {i: b[i]["verdict"] == "correct" for i in ids}
    only_a = sum(1 for i in ids if a_ok[i] and not b_ok[i])
    only_b = sum(1 for i in ids if b_ok[i] and not a_ok[i])
    disc = only_a + only_b

    delta = (only_b - only_a) / n
    # McNemar CI on the paired difference
    if disc:
        var = (disc - (only_b - only_a) ** 2 / n) / n ** 2
        half = Z_ALPHA * math.sqrt(max(var, 0.0))
    else:
        half = 0.0
    mdd = (Z_ALPHA + Z_BETA) * math.sqrt((disc / n) / n) if disc else 0.0

    return {
        "n": n,
        "acc_a": sum(a_ok.values()) / n,
        "acc_b": sum(b_ok.values()) / n,
        "delta": delta,
        "ci": (delta - half, delta + half),
        "only_a": only_a,
        "only_b": only_b,
        "discordant": disc,
        "discordance_rate": disc / n,
        "p": mcnemar_p(only_a, only_b),
        "mdd": mdd,
        "significant": mcnemar_p(only_a, only_b) < 0.05,
        "underpowered": abs(delta) < mdd,
    }


def report(name, r):
    print(f"\n  {name}  (n = {r['n']:,d})")
    print(f"    A {r['acc_a'] * 100:5.2f}%    B {r['acc_b'] * 100:5.2f}%    "
          f"delta {r['delta'] * 100:+.2f} pp   "
          f"95% CI [{r['ci'][0] * 100:+.2f}, {r['ci'][1] * 100:+.2f}]")
    print(f"    discordant {r['discordant']:,d} "
          f"({r['discordance_rate'] * 100:.1f}%)   "
          f"only-A {r['only_a']:,d}   only-B {r['only_b']:,d}")
    print(f"    McNemar p {r['p']:.4g}    MDD at this discordance "
          f"{r['mdd'] * 100:.2f} pp")
    if r["significant"]:
        note = ""
        if r["underpowered"]:
            note = ("  (below the 80%-power threshold, so an effect this "
                    "size would often be missed - but it was found here)")
        print(f"    -> SIGNIFICANT, CI excludes zero{note}")
    else:
        print(f"    -> not significant; powered to find "
              f"{r['mdd'] * 100:.2f} pp or more, so a smaller true "
              f"difference would likely be missed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    args = ap.parse_args()

    a, b = load_run(args.run_a), load_run(args.run_b)
    print(f"A = {args.run_a}\nB = {args.run_b}")

    report("full test", compare(a, b))
    report("clean subset", compare(a, b, subset=clean_ids()))

    # Where the two runs disagree, and on what
    ids = sorted(set(a) & set(b))
    flips = [(i, a[i], b[i]) for i in ids
             if (a[i]["verdict"] == "correct") != (b[i]["verdict"] == "correct")]
    if flips:
        print(f"\n  sample disagreements ({len(flips):,d} total)")
        for i, ra, rb in flips[:6]:
            print(f"    {i}  gold {ra['gold']}")
            print(f"      A [{ra['verdict']:11s}] {str(ra.get('pred'))[:34]:34s}"
                  f"  B [{rb['verdict']:11s}] {str(rb.get('pred'))[:34]}")


if __name__ == "__main__":
    main()
