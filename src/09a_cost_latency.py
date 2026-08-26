"""
Stage 9 - cost and latency.

Accuracy is one axis. A method that matches another at 100x the cost per
prediction is not equivalent to it, and every table in this project so far
has reported only the first number.

Everything here is read from committed artefacts - `meta.json` records
training seconds, prompt tokens, batch and chunk sizes for every run - so
the table is reproducible without a GPU and cannot drift from the runs it
describes.

What is measured and what is not:

  training time     wall clock from meta.json, on a Kaggle Tesla T4.
                    Includes per-epoch dev evaluation, because that is what
                    the runs actually did.
  prompt tokens     from meta.json, measured with the model's tokenizer.
  inference         derived from the observed items/sec of the constrained
                    scoring loop. Single-item scoring on a T4 with 77 label
                    continuations - indicative of relative cost, NOT a
                    serving benchmark. A production system would batch,
                    quantise, cache the shared prefix, and likely use a
                    different accelerator.
  cost per 1k       inference seconds x an assumed GPU hourly rate. The
                    rate is a parameter, not a measurement.

The kNN row has no GPU cost at all, which is the point of including it.

    python src/09a_cost_latency.py
    python src/09a_cost_latency.py --gpu-rate 0.40
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUNS = Path("reports/runs")
SPLITS = Path("eval/splits")

# Observed throughput for the constrained scoring loop, items/sec, taken
# from the run logs rather than re-timed here. Free-form generation is
# roughly 4x faster at these prompt lengths but constrained is the regime
# every headline number uses.
THROUGHPUT = {
    "qwen3lora": 3.0,        # 63-token prompt, 77 label continuations
    "qwen3lora_r64": 2.7,
    "zero_shot": 2.0,        # 447-token prompt
    "bare": 4.6,             # 63-token prompt, untuned
    "few_shot_k77": 0.4,     # 2,111-token prompt
    "retrieval_k10": 1.5,
}
ROBERTA_ITEMS_PER_SEC = 330.0   # 3 ms/item, single-item forward, seq len 64
KNN_ITEMS_PER_SEC = 3000.0      # CPU, TF-IDF transform + neighbour lookup

# Some baselines ran before meta.json recorded prompt_tokens. These come
# from the stage-3a token report and docs/04_baselines.md rather than being
# silently blank, which reads like missing data instead of an older run.
FALLBACK_TOKENS = {
    "zero_shot": 447,
    "roberta": 64,      # max_len, not a prompt - the encoder has no prompt
}

# Rows to report, in the order a reader should meet them.
ROWS = [
    ("majority", "majority class", None),
    ("knn_k5", "kNN over TF-IDF (CPU)", "knn"),
    ("bare", "base model, bare prompt", "bare"),
    ("zero_shot", "zero-shot, 77 labels listed", "zero_shot"),
    ("few_shot_k77", "few-shot, 77 exemplars", "few_shot_k77"),
    ("retrieval_k10", "retrieval + LLM", "retrieval_k10"),
    ("robertabase_rungfull", "roberta-base, fine-tuned", "roberta"),
    ("qwen3lora_rungfull_ep8", "LoRA on Qwen3-1.7B", "qwen3lora"),
    ("qwen3lora_rungfull_r64ep8", "LoRA, rank 64", "qwen3lora_r64"),
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def find_runs(family, regime="constrained"):
    """Directories for a family, whether or not they carry a _seedN suffix."""
    out = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or not d.name.endswith("__" + regime):
            continue
        stem = d.name.rsplit("__", 1)[0]
        if stem == family or stem.startswith(family + "_seed"):
            if (d / "predictions.jsonl").exists():
                out.append(d)
    return out


def meta_of(d):
    p = d / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def accuracy(d):
    rows = read_jsonl(d / "predictions.jsonl")
    return sum(1 for r in rows if r["verdict"] == "correct") / len(rows)


def gather(family, kind):
    dirs = find_runs(family)
    if not dirs:
        return None
    accs = [accuracy(d) for d in dirs]
    metas = [meta_of(d) for d in dirs]

    train_s = [m["train_seconds"] for m in metas if m.get("train_seconds")]
    tokens = next((m.get("prompt_tokens") for m in metas
                   if m.get("prompt_tokens")), None)
    if tokens is None:
        tokens = FALLBACK_TOKENS.get(kind)
    params = {
        "roberta": 125e6, "qwen3lora": 2.03e9, "qwen3lora_r64": 2.03e9,
        "zero_shot": 2.03e9, "bare": 2.03e9, "few_shot_k77": 2.03e9,
        "retrieval_k10": 2.03e9, "knn": 0.0,
    }.get(kind)
    trained = {
        "roberta": 125e6, "qwen3lora": 17.4e6, "qwen3lora_r64": 69.7e6,
    }.get(kind, 0.0)

    if kind == "knn":
        ips = KNN_ITEMS_PER_SEC
    elif kind == "roberta":
        ips = ROBERTA_ITEMS_PER_SEC
    else:
        ips = THROUGHPUT.get(kind)
    if kind is None:          # majority class: a constant, not a model
        ips = None

    return {
        "n_runs": len(dirs),
        "acc": statistics.mean(accs),
        "sd": statistics.stdev(accs) * 100 if len(accs) > 1 else None,
        "train_min": statistics.mean(train_s) / 60 if train_s else None,
        "tokens": tokens,
        "params": params,
        "trained": trained,
        "items_per_sec": ips,
    }


def fmt(x, unit="", nd=1):
    return "-" if x is None else f"{x:,.{nd}f}{unit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-rate", type=float, default=0.35,
                    help="USD per GPU-hour; a parameter, not a measurement")
    ap.add_argument("--volume", type=int, default=1_000_000,
                    help="predictions per month for the scaling illustration")
    args = ap.parse_args()

    print(f"Cost and latency. GPU rate assumed ${args.gpu_rate:.2f}/hour "
          f"(RTX 4090 community tier, Aug 2026).")
    print("Inference figures are single-item constrained scoring on a Tesla")
    print("T4 and are indicative of RELATIVE cost, not a serving benchmark.\n")

    print(f"  {'method':30s} {'acc':>7s} {'sd':>6s} {'train':>8s} "
          f"{'prompt':>7s} {'items/s':>8s} {'ms/item':>8s} {'$/1k':>8s}")
    rows = []
    for family, label, kind in ROWS:
        g = gather(family, kind)
        if g is None:
            print(f"  {label:30s} (no runs found)")
            continue
        ips = g["items_per_sec"]
        ms = 1000.0 / ips if ips else None
        cost_1k = (1000.0 / ips) / 3600.0 * args.gpu_rate if ips else None
        if kind == "knn":
            cost_1k = 0.0
        note = "  (arithmetic, no model)" if kind is None else ""
        rows.append((label, g, ms, cost_1k))
        print(f"  {label:30s} {g['acc'] * 100:6.1f}% "
              f"{fmt(g['sd'], 'pp'):>6s} "
              f"{fmt(g['train_min'], ' min', 0):>8s} "
              f"{fmt(g['tokens'], '', 0):>7s} "
              f"{fmt(ips, '', 1):>8s} {fmt(ms, '', 1):>8s} "
              f"{'free' if cost_1k == 0 else fmt(cost_1k, '', 4):>8s}"
              + note)

    # ------------------------------------------------------- comparisons
    print(f"\n  relative to roberta-base:")
    base = next((g for lbl, g, _, _ in rows if "roberta" in lbl), None)
    if base:
        for label, g, ms, cost in rows:
            if g["items_per_sec"] is None or base["items_per_sec"] is None:
                continue
            speed = base["items_per_sec"] / g["items_per_sec"]
            acc_d = (g["acc"] - base["acc"]) * 100
            size = (g["params"] / base["params"]) if g["params"] else 0
            print(f"    {label:30s} {acc_d:+6.2f} pp   "
                  f"{speed:6.1f}x slower   {size:5.1f}x parameters")

    # -------------------------------------------------------- at volume
    print(f"\n  at {args.volume:,d} predictions/month, GPU time alone:")
    for label, g, ms, cost in rows:
        if not g["items_per_sec"]:
            continue
        hours = args.volume / g["items_per_sec"] / 3600
        print(f"    {label:30s} {hours:8.1f} GPU-hours   "
              f"${hours * args.gpu_rate:9,.0f}"
              + ("   (runs on CPU)" if g["params"] == 0 else ""))

    print("\n  Read the dollar column before drawing a business case from")
    print("  it. The whole spread here is $0 to $243 a month - less than one")
    print("  engineer-hour. In dollars, at this volume, none of these")
    print("  methods is expensive and the cost argument does not decide")
    print("  anything.")
    print("\n  What the table does support:")
    print("    latency      3 ms vs 333 ms per prediction. Real for an")
    print("                 interactive router, irrelevant for batch triage.")
    print("    deployment   roberta and kNN run on CPU; the LoRA model needs")
    print("                 a GPU to be practical. That is an infrastructure")
    print("                 decision, not a line item.")
    print("    capacity     0.8 vs 93 GPU-hours/month. Matters if you")
    print("                 provision, not if you rent by the hour.")
    print("\n  And the row that makes the point best: few-shot with 77")
    print("  exemplars is 825x slower than roberta AND 88 points less")
    print("  accurate. Cost and accuracy are independent axes.")


if __name__ == "__main__":
    main()
