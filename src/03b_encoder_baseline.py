"""
Stage 4 - the encoder baseline.

The comparison most write-ups skip. DeBERTa-v3-base is ~184M parameters
against Qwen3-1.7B's 2.03B - roughly 11x smaller - and it is trained the
old way: a classification head over the whole model, no LoRA, no prompt,
no generation. It cannot invent a label, so `unparseable` cannot occur and
only the constrained regime applies.

Trained on the same five nested rungs the LoRA will use, so the two produce
scaling curves on identical axes and the question "does the small
conventional model need less data than the fine-tuned LLM?" is answerable
rather than assumed.

Every run also gets three seeds, because a training seed is a noise source
with no analogue in the prompting baselines and stage 5 needs the number.
Stage 0's sizing simulation predicted that seed variance, not test-set
size, would be the binding constraint on this project - this is the first
chance to measure it rather than simulate it.

Nothing here is tuned against test. Hyperparameters are fixed before the
first run and the same for every rung; model selection within a run uses
dev.

    python src/03b_encoder_baseline.py --rungs 8 --seeds 1     # smoke
    python src/03b_encoder_baseline.py                          # everything
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()   # the per-layer load report is 200 lines

from evaluator import evaluate_constrained, score  # noqa: E402
from runner import write_env  # noqa: E402

SPLITS = Path("eval/splits")
RUNS = Path("reports/runs")
MODEL = "microsoft/deberta-v3-base"

# Fixed before the first run, identical for every rung and seed. Chosen
# from the model card's usual range, not tuned here - tuning per rung
# would confound "more data helps" with "this rung got better settings".
LR = 2e-5
# Fixed epochs means the smallest rung sees ~200 gradient steps and the
# largest ~2,300, so the scaling curve partly measures optimisation budget
# rather than data quantity. Best-epoch-on-dev mitigates this - each run
# stops where dev says it should rather than where the schedule ends - but
# the confound is real and is stated in docs/04_encoder.md rather than
# engineered away. Fixed *steps* would remove it and introduce a different
# one: the largest rung would see each example far fewer times.
EPOCHS = 20
BATCH = 16
MAX_LEN = 64          # p95 of Banking77 is 29 words; 64 tokens is ample
WARMUP_RATIO = 0.1
SEEDS = [1, 2, 3]
RUNGS = [2, 4, 8, 16, 24]   # per class -> 154 / 308 / 616 / 1232 / 1848


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_ids():
    m = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    thr = m["near_dup_threshold_recorded"]
    return [r["id"] for r in read_jsonl(SPLITS / "near_dup_test.jsonl")
            if r["twin_sim"] < thr]


def load_rung(k, pool_by_id):
    ids = json.loads((SPLITS / "rungs" / f"rung_{k:02d}.json")
                     .read_text(encoding="utf-8"))
    return [pool_by_id[i] for i in ids]


def make_dataset(rows, labels, tok):
    from torch.utils.data import TensorDataset
    idx = {lbl: i for i, lbl in enumerate(labels)}
    enc = tok([r["text"] for r in rows], truncation=True, max_length=MAX_LEN,
              padding="max_length", return_tensors="pt")
    y = torch.tensor([idx[r["label"]] for r in rows])
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for ids, mask, _ in loader:
        logits = model(input_ids=ids.to(device),
                       attention_mask=mask.to(device)).logits
        out.append(logits.argmax(-1).cpu())
    return torch.cat(out).tolist()


def train_one(rung_rows, dev_rows, labels, seed, device):
    """One training run. Best epoch chosen on dev, never on test."""
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)

    torch.manual_seed(seed)
    np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    # dtype=float32 is load-bearing. Without it transformers takes the
    # dtype from the checkpoint and loads fp16, and DeBERTa-v3's
    # disentangled attention overflows in half precision: loss starts
    # correctly at ln(77) and is NaN within ten steps, at every learning
    # rate. Fifteen runs once trained to exactly 1/77 for this reason.
    # Qwen is fine in fp16; this model is not.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels), dtype=torch.float32).to(device)
    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {torch.float32}:
        raise RuntimeError(f"expected fp32 parameters, got {dtypes}")

    train_ds = make_dataset(rung_rows, labels, tok)
    dev_ds = make_dataset(dev_rows, labels, tok)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    dev_dl = DataLoader(dev_ds, batch_size=64)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total = len(train_dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(
        opt, int(total * WARMUP_RATIO), total)

    idx = {lbl: i for i, lbl in enumerate(labels)}
    dev_y = [idx[r["label"]] for r in dev_rows]
    best_acc, best_state, best_epoch = -1.0, None, -1

    history = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = []
        for ids, mask, y in train_dl:
            opt.zero_grad()
            loss = model(input_ids=ids.to(device),
                         attention_mask=mask.to(device),
                         labels=y.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            epoch_loss.append(loss.item())

        mean_loss = float(np.mean(epoch_loss))
        # A run that never learns is invisible in a final accuracy number.
        # Fifteen runs once trained to exactly chance and reported only
        # "1.3%", because this line did not exist.
        if not np.isfinite(mean_loss):
            raise RuntimeError(
                f"loss is {mean_loss} at epoch {epoch} - training diverged")

        acc = np.mean(np.array(predict(model, dev_dl, device)) == dev_y)
        history.append({"epoch": epoch, "loss": round(mean_loss, 4),
                        "dev_acc": round(float(acc), 4)})
        if epoch < 3 or epoch == EPOCHS - 1:
            print(f"      ep{epoch:<2d} loss {mean_loss:6.3f}  "
                  f"dev {acc * 100:5.1f}%", flush=True)
        if acc > best_acc:
            best_acc, best_epoch = float(acc), epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # Chance is 1/77 = 1.3%. Finishing there means the run is worthless and
    # must not be silently written out as a data point.
    if best_acc <= 2.0 / len(labels):
        print(f"      !! dev accuracy {best_acc * 100:.1f}% is at chance "
              f"(1/{len(labels)} = {100 / len(labels):.1f}%) - "
              f"first-epoch loss {history[0]['loss']:.3f}, "
              f"last {history[-1]['loss']:.3f}")

    return model, tok, {"dev_acc": best_acc, "best_epoch": best_epoch,
                        "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--rungs", nargs="*", type=int, default=RUNGS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = read_jsonl(SPLITS / "train_pool.jsonl")
    pool_by_id = {r["id"]: r for r in pool}
    dev = read_jsonl(SPLITS / "dev.jsonl")
    test = read_jsonl(SPLITS / "test.jsonl")
    labels = sorted({r["label"] for r in pool})
    clean = clean_ids()

    print(f"{MODEL} on {device}   rungs {args.rungs}   seeds {args.seeds}")
    print(f"test {len(test):,d}   dev {len(dev):,d}   clean {len(clean):,d}")

    from torch.utils.data import DataLoader
    summary = []

    for k in args.rungs:
        rung = load_rung(k, pool_by_id)
        for seed in args.seeds:
            name = f"deberta_rung{k:02d}_seed{seed}"
            d = runs / f"{name}__constrained"
            if (d / "predictions.jsonl").exists():
                recs = read_jsonl(d / "predictions.jsonl")
                s = score(recs)
                print(f"  {name:30s} already done  {s['accuracy'] * 100:5.1f}%")
                summary.append((k, seed, s["accuracy"],
                                score(recs, subset_ids=clean)["accuracy"],
                                None, None))
                continue

            t0 = time.time()
            model, tok, info = train_one(rung, dev, labels, seed, device)
            train_s = time.time() - t0

            test_dl = DataLoader(make_dataset(test, labels, tok),
                                 batch_size=64)
            preds = [labels[i] for i in predict(model, test_dl, device)]
            recs = [evaluate_constrained(p, r["label"], labels)
                    | {"id": r["id"], "query": r["text"]}
                    for p, r in zip(preds, test)]

            d.mkdir(parents=True, exist_ok=True)
            write_env(d)
            with open(d / "predictions.jsonl", "w", encoding="utf-8",
                      newline="\n") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False,
                                       sort_keys=True) + "\n")
            (d / "meta.json").write_text(json.dumps({
                "config": name, "regime": "constrained", "model": MODEL,
                "rung_per_class": k, "n_train": len(rung), "seed": seed,
                "lr": LR, "epochs": EPOCHS, "batch": BATCH,
                "max_len": MAX_LEN, "best_epoch": info["best_epoch"],
                "dev_acc": info["dev_acc"], "train_seconds": round(train_s),
                "history": info["history"],
                "n_items": len(test),
            }, indent=2), encoding="utf-8", newline="\n")

            s = score(recs)
            c = score(recs, subset_ids=clean)
            print(f"  {name:30s} test {s['accuracy'] * 100:5.1f}%  "
                  f"clean {c['accuracy'] * 100:5.1f}%  "
                  f"dev {info['dev_acc'] * 100:5.1f}%  "
                  f"ep{info['best_epoch']}  {train_s / 60:.1f} min")
            summary.append((k, seed, s["accuracy"], c["accuracy"],
                            info["dev_acc"], train_s))

            del model
            torch.cuda.empty_cache()

    print(f"\n{'=' * 68}\nscaling curve, mean over seeds\n{'=' * 68}")
    print(f"  {'per class':>9s} {'n_train':>8s} {'test':>8s} {'clean':>8s} "
          f"{'seed sd':>8s} {'range':>8s}")
    for k in args.rungs:
        vals = [s[2] for s in summary if s[0] == k]
        cl = [s[3] for s in summary if s[0] == k]
        if not vals:
            continue
        sd = float(np.std(vals, ddof=1)) * 100 if len(vals) > 1 else float("nan")
        rng = (max(vals) - min(vals)) * 100 if len(vals) > 1 else float("nan")
        n_train = sum(1 for _ in load_rung(k, pool_by_id))
        print(f"  {k:9d} {n_train:8,d} {np.mean(vals) * 100:7.1f}% "
              f"{np.mean(cl) * 100:7.1f}% {sd:7.2f}pp {rng:7.2f}pp")

    print("\n  'seed sd' is the training-seed noise stage 0 predicted would")
    print("  bind this project. Every later delta is judged against it.")
    print("  Bar to clear: kNN k=5 at 82.9% +/- 1.3, CPU, no training.")


if __name__ == "__main__":
    main()
