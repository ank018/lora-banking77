"""
Stage 6 - ablations.

One factor varied at a time from a centre configuration fixed before any
LoRA run happened: r=16, alpha=32, dropout 0.05, lr 2e-4, targeting the
four attention projections plus the MLP triple. Because the centre was
chosen in advance, these are honest variations rather than a hyperparameter
search presented afterwards as an ablation.

Scored on DEV, not test. Rank and learning rate are choices, and choices
are made on dev - that is what dev is for. It is also 5x cheaper: 616 items
by free-form generation takes seconds, against 19 minutes for a constrained
pass over 3,080. Test is touched once, at the end, and only if a
configuration is adopted.

  rank            8, 16, 64      alpha scales with rank to hold alpha/r
                                 fixed; otherwise varying rank silently
                                 varies the effective learning rate of the
                                 adapter and the ablation measures two
                                 things at once
  learning rate   1e-4, 2e-4, 5e-4
  targets         attention only vs attention + MLP

Every configuration runs three seeds, because stage 4 measured seed noise
at 0.23-0.26 pp and a single run cannot distinguish a 1-point effect from
a lucky draw. With three seeds the standard error of a mean is about
0.15 pp, so differences near half a point become visible.

Written prediction, before any of these run:
  rank        differences inside noise; r=8 no worse than r=64
  lr          matters; 5e-4 unstable or worse, 1e-4 undertrained at this
              epoch budget
  targets     attention-only within ~1 point of attention+MLP

    python src/05_ablations.py --rung 8 --seeds 1 2 3
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

SPLITS = Path("eval/splits")
REPORTS = Path("reports")

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]

# name -> overrides applied to the centre configuration
ABLATIONS = {
    "centre":        {},
    "rank_8":        {"RANK": 8, "ALPHA": 16},
    "rank_64":       {"RANK": 64, "ALPHA": 128},
    "lr_1e-4":       {"LR": 1e-4},
    "lr_5e-4":       {"LR": 5e-4},
    "attn_only":     {"TARGETS": ATTN},
}


def load_trainer():
    """04_train_lora.py starts with a digit, so it cannot be imported
    normally. Reusing it rather than copying the training loop means the
    ablations exercise exactly the code the headline runs used."""
    path = Path(__file__).resolve().parent / "04_train_lora.py"
    spec = importlib.util.spec_from_file_location("_lora", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=8,
                    help="per-class rung to ablate on; 0 for the full pool")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=list(ABLATIONS))
    ap.add_argument("--out", default=str(REPORTS / "ablations.json"))
    args = ap.parse_args()

    L = load_trainer()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(L.MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    pool = read_jsonl(SPLITS / "train_pool.jsonl")
    pool_by_id = {r["id"]: r for r in pool}
    dev = read_jsonl(SPLITS / "dev.jsonl")
    labels = sorted({r["label"] for r in pool})
    train_rows = L.load_rung(args.rung, pool_by_id)

    centre = {"RANK": L.RANK, "ALPHA": L.ALPHA, "LR": L.LR,
              "TARGETS": list(L.TARGETS), "DROPOUT": L.DROPOUT}
    print(f"{L.MODEL}  rung {args.rung} ({len(train_rows):,d} examples)  "
          f"dev {len(dev)}  epochs {args.epochs}  seeds {args.seeds}")
    print(f"centre: r={centre['RANK']} alpha={centre['ALPHA']} "
          f"lr={centre['LR']} targets={len(centre['TARGETS'])} modules")
    print("scored on dev, free-form; test is not touched\n")

    out_path = Path(args.out)
    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))

    for name in args.only:
        overrides = ABLATIONS[name]
        done = [s for s in args.seeds
                if str(s) in results.get(name, {}).get("seeds", {})]
        if len(done) == len(args.seeds):
            print(f"  {name}: already complete")
            continue

        for key, val in centre.items():          # reset, then apply
            setattr(L, key, val)
        for key, val in overrides.items():
            setattr(L, key, val)

        desc = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "as fixed"
        print(f"  {name:12s} {desc}")

        entry = results.setdefault(name, {"overrides": {
            k: (v if not isinstance(v, list) else len(v))
            for k, v in overrides.items()}, "seeds": {}})
        for seed in args.seeds:
            if str(seed) in entry["seeds"]:
                continue
            t0 = time.time()
            model, info = L.train_one(train_rows, dev, labels, seed, tok,
                                      args.epochs)
            entry["seeds"][str(seed)] = {
                "dev_acc": info["dev_acc"], "best_epoch": info["best_epoch"],
                "trainable": info["trainable"],
                "minutes": round((time.time() - t0) / 60, 1),
                "history": info["history"],
            }
            print(f"      seed {seed}: dev {info['dev_acc'] * 100:5.2f}%  "
                  f"ep{info['best_epoch']}  "
                  f"{info['trainable'] / 1e6:.1f}M trainable  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            del model
            torch.cuda.empty_cache()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2),
                                encoding="utf-8", newline="\n")

    # ------------------------------------------------------------ report
    print(f"\n{'=' * 68}\nablation summary (dev accuracy)\n{'=' * 68}")
    print(f"  {'config':12s} {'dev mean':>9s} {'sd':>7s} {'vs centre':>10s} "
          f"{'trainable':>10s} {'min/run':>8s}")

    def stats(name):
        seeds = results.get(name, {}).get("seeds", {})
        accs = [v["dev_acc"] for v in seeds.values()]
        return accs, seeds

    base_accs, _ = stats("centre")
    base_mean = statistics.mean(base_accs) if base_accs else None

    for name in args.only:
        accs, seeds = stats(name)
        if not accs:
            continue
        mean = statistics.mean(accs)
        sd = statistics.stdev(accs) * 100 if len(accs) > 1 else float("nan")
        delta = (mean - base_mean) * 100 if base_mean is not None else 0.0
        tr = next(iter(seeds.values()))["trainable"] / 1e6
        mins = statistics.mean(v["minutes"] for v in seeds.values())
        print(f"  {name:12s} {mean * 100:8.2f}% {sd:6.2f}pp "
              f"{delta:+9.2f} {tr:9.1f}M {mins:7.1f}")

    print("\n  Seed noise on this task is 0.23-0.26 pp (stage 4), so the SE")
    print("  of a 3-seed mean is about 0.15 pp and a difference of two means")
    print("  needs roughly 0.6 pp to be worth calling real. Anything smaller")
    print("  is a draw, and should be reported as one.")


if __name__ == "__main__":
    main()
