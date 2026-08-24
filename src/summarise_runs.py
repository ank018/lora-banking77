"""
The definitive results table, computed from committed artefacts.

Each stage script prints a summary at the end of its own run, which sees
only the runs produced in that session. A rerun of two of three seeds
therefore prints a confident standard deviation over two seeds - that
happened, and reported 0.05 pp where the correct figure across all three
was 0.26 pp.

This reads reports/runs/ instead. Every number in the README comes from
here, so there is one place where the aggregation logic lives and one
place to check it.

Multi-seed configurations are grouped and reported as mean, standard
deviation and range. Single-run configurations are reported with a
binomial CI. Both are scored on the full test set and on the
near-duplicate-free subset.

    python src/summarise_runs.py
    python src/summarise_runs.py --filter roberta
"""

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluator import score  # noqa: E402

SPLITS = Path("eval/splits")
RUNS = Path("reports/runs")
SEED_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_ids():
    m = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    thr = m["near_dup_threshold_recorded"]
    return [r["id"] for r in read_jsonl(SPLITS / "near_dup_test.jsonl")
            if r["twin_sim"] < thr]


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n) * 100


def collect(runs, clean):
    """One entry per run directory: (family, seed, n, acc, clean_acc)."""
    out = []
    for d in sorted(runs.iterdir()):
        pred = d / "predictions.jsonl"
        if not d.is_dir() or not pred.exists():
            continue
        rows = read_jsonl(pred)
        if not rows:
            continue
        name = d.name
        regime = "constrained" if name.endswith("__constrained") else (
            "free_form" if name.endswith("__free_form") else "?")
        config = name.rsplit("__", 1)[0]
        m = SEED_RE.match(config)
        family, seed = (m.group("base"), int(m.group("seed"))) if m else (
            config, None)

        meta = {}
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))

        s = score(rows)
        c = score(rows, subset_ids=clean)
        out.append({
            "family": family, "regime": regime, "seed": seed,
            "n": s["n"], "acc": s["accuracy"], "clean": c["accuracy"],
            "unparseable": s["unparseable_rate"],
            "n_train": meta.get("n_train") or meta.get("n_pool"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--filter", default=None,
                    help="only families containing this substring")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    clean = clean_ids()
    entries = collect(runs, clean)
    if args.filter:
        entries = [e for e in entries if args.filter in e["family"]]
    if not entries:
        print("no runs found")
        return

    groups = {}
    for e in entries:
        groups.setdefault((e["family"], e["regime"]), []).append(e)

    print(f"{len(entries)} runs in {len(groups)} configurations   "
          f"clean subset n={len(clean):,d}")
    print(f"\n  {'configuration':34s} {'regime':11s} {'n_tr':>6s} "
          f"{'seeds':>5s} {'test':>8s} {'clean':>8s} {'sd':>7s} {'range':>7s}")

    for (family, regime), runs_in in sorted(groups.items()):
        accs = [r["acc"] for r in runs_in]
        cleans = [r["clean"] for r in runs_in]
        n_tr = runs_in[0]["n_train"]
        n_tr_s = f"{n_tr:,}" if n_tr else "-"
        mean, cmean = statistics.mean(accs), statistics.mean(cleans)

        if len(accs) > 1:
            sd = statistics.stdev(accs) * 100
            rng = (max(accs) - min(accs)) * 100
            tail = f" {sd:6.2f}pp {rng:6.2f}pp"
        else:
            tail = f" {'+/-' + f'{ci95(mean, runs_in[0]['n']):.1f}':>8s} "\
                   f"{'(CI)':>8s}"
        print(f"  {family:34s} {regime:11s} {n_tr_s:>6s} "
              f"{len(accs):5d} {mean * 100:7.1f}% {cmean * 100:7.1f}%{tail}")

    print("\n  sd and range are across training seeds where more than one")
    print("  exists; otherwise a 95% binomial CI on the single run.")
    print("  Seed sd is the noise floor every ablation delta is judged")
    print("  against. Use src/compare.py for any two-way comparison - a")
    print("  paired McNemar test is tighter than these unpaired bars.")


if __name__ == "__main__":
    main()
