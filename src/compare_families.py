"""
Compare two training procedures, not two runs.

compare.py answers "did these two prediction files differ", which is the
right question for a single run against a single run. It is the wrong
question for "is roberta better than LoRA", because each of those is a
procedure that produces a distribution of models - and a single draw from
each throws away the seed variance we paid three hours to measure.

Two tests, reported together because they answer different things:

  seed-level    Welch t-test on the per-seed accuracies. Treats each
                trained model as one observation of the procedure. Accounts
                for training variance; ignores which items were missed.
                Low power with three seeds a side - df is about 4 - so a
                null result here is weak evidence.

  item-level    McNemar on each matched seed pair. Uses every one of the
                3,080 items, and is far tighter, but conditions on the
                particular models drawn.

Agreement between them is the interesting case. Disagreement usually means
the effect is smaller than seed noise, and should be reported as such
rather than as whichever test gave the more satisfying answer.

    python src/compare_families.py robertabase_rungfull qwen3lora_rungfull
"""

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare import compare, load_run  # noqa: E402

SPLITS = Path("eval/splits")
RUNS = Path("reports/runs")
SEED_RE = re.compile(r"_seed(\d+)__")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_ids():
    m = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    thr = m["near_dup_threshold_recorded"]
    return {r["id"] for r in read_jsonl(SPLITS / "near_dup_test.jsonl")
            if r["twin_sim"] < thr}


def find_seeds(runs, family, regime):
    out = {}
    for d in sorted(runs.iterdir()):
        if not d.is_dir() or not d.name.startswith(family + "_seed"):
            continue
        if not d.name.endswith("__" + regime):
            continue
        m = SEED_RE.search(d.name)
        if m and (d / "predictions.jsonl").exists():
            out[int(m.group(1))] = d
    return out


def accuracy(run, subset=None):
    rows = run.values()
    if subset is not None:
        rows = [r for i, r in run.items() if i in subset]
    rows = list(rows)
    return sum(1 for r in rows if r["verdict"] == "correct") / len(rows)


def welch(a, b):
    """Welch's t-test. Returns (t, df, p) using a normal-tail approximation
    for p, which is conservative-ish at these tiny df and is flagged as
    such rather than dressed up with a precise-looking number."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return None
    t = (mb - ma) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
    except Exception:  # noqa: BLE001
        p = 2 * 0.5 * math.erfc(abs(t) / math.sqrt(2))
    return t, df, p, se


def report_family(name, accs):
    if len(accs) > 1:
        print(f"    {name:26s} {statistics.mean(accs) * 100:6.2f}%  "
              f"sd {statistics.stdev(accs) * 100:.2f}pp  "
              f"n={len(accs)}  [" +
              ", ".join(f"{a * 100:.2f}" for a in sorted(accs)) + "]")
    else:
        print(f"    {name:26s} {accs[0] * 100:6.2f}%  single run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family_a")
    ap.add_argument("family_b")
    ap.add_argument("--regime", default="constrained")
    ap.add_argument("--runs-dir", default=str(RUNS))
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    a_dirs = find_seeds(runs, args.family_a, args.regime)
    b_dirs = find_seeds(runs, args.family_b, args.regime)
    if not a_dirs or not b_dirs:
        print(f"no runs found for {args.family_a} / {args.family_b} "
              f"({args.regime})")
        return

    a_runs = {s: load_run(d) for s, d in a_dirs.items()}
    b_runs = {s: load_run(d) for s, d in b_dirs.items()}
    clean = clean_ids()

    print(f"A = {args.family_a}   seeds {sorted(a_dirs)}")
    print(f"B = {args.family_b}   seeds {sorted(b_dirs)}")
    print(f"regime: {args.regime}")

    for tag, subset in (("full test", None), ("clean subset", clean)):
        a_acc = [accuracy(r, subset) for r in a_runs.values()]
        b_acc = [accuracy(r, subset) for r in b_runs.values()]
        print(f"\n  {tag}")
        report_family(args.family_a, a_acc)
        report_family(args.family_b, b_acc)

        delta = (statistics.mean(b_acc) - statistics.mean(a_acc)) * 100
        w = welch(a_acc, b_acc)
        if w:
            t, df, p, se = w
            half = 1.96 * se * 100
            print(f"    seed-level   delta {delta:+.2f} pp   "
                  f"SE {se * 100:.2f}   t {t:+.2f}  df {df:.1f}  p {p:.3f}")
            print(f"                 95% CI [{delta - half:+.2f}, "
                  f"{delta + half:+.2f}] pp")
            verdict = ("B differs from A" if p < 0.05
                       else "no significant difference between procedures")
            print(f"                 -> {verdict}")
        else:
            print(f"    seed-level   delta {delta:+.2f} pp  "
                  f"(too few seeds for a test)")

    print("\n  item-level, matched seeds (McNemar on all 3,080 items)")
    shared = sorted(set(a_dirs) & set(b_dirs))
    if not shared:
        print("    no matching seed numbers")
    for s in shared:
        r = compare(a_runs[s], b_runs[s])
        flag = "significant" if r["significant"] else "n.s."
        print(f"    seed {s}: delta {r['delta'] * 100:+.2f} pp   "
              f"discordant {r['discordant']:,d}   "
              f"p {r['p']:.4f}   {flag}")

    print("\n  The seed-level test compares procedures and has low power at")
    print("  three seeds. The item-level tests are tighter but condition on")
    print("  the particular models drawn. Report both; where they disagree,")
    print("  the effect is comparable to seed noise and should be described")
    print("  that way rather than by whichever test reads better.")


if __name__ == "__main__":
    main()
