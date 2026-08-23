"""
Why did DeBERTa train to exactly chance?

All 15 runs scored 1.3% = 1/77 on test and dev, at every rung, every seed,
with best epoch 0. That is a constant prediction: the model learned
nothing. Candidates, none yet ruled out:

  loss is NaN/inf        weights become NaN, argmax collapses to one class
  tokenizer is broken    inputs are garbage, so there is nothing to learn
  wrong dtype            fp16 - DeBERTa-v3 is known to overflow in fp16
  labels misaligned      targets do not correspond to inputs
  optimiser not stepping  parameters never move

The main script printed no training loss, so 15 runs produced no evidence
about any of this. That gap is the actual bug; the training failure is
downstream of it.

This runs ~40 steps on the smallest rung and prints everything the main
script should have been printing all along.

    python src/03e_diagnose_encoder.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

SPLITS = Path("eval/splits")
MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 64
BATCH = 16
LRS = [2e-5, 5e-5, 1e-4]
STEPS = 40


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rule(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


def main():
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = {r["id"]: r for r in read_jsonl(SPLITS / "train_pool.jsonl")}
    ids = json.loads((SPLITS / "rungs" / "rung_08.json").read_text())
    rows = [pool[i] for i in ids]
    labels = sorted({r["label"] for r in pool.values()})
    idx = {lbl: i for i, lbl in enumerate(labels)}

    # ---------------------------------------------------------------- 1
    rule("1. tokenizer")
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"  class        {tok.__class__.__name__}")
    print(f"  vocab size   {len(tok)}")
    sample = rows[0]["text"]
    enc = tok(sample, truncation=True, max_length=MAX_LEN)
    print(f"  text         {sample!r}")
    print(f"  ids          {enc['input_ids'][:16]}")
    print(f"  decoded      {tok.decode(enc['input_ids'])!r}")
    if len(enc["input_ids"]) <= 2:
        print("  !! tokenised to nothing but special tokens")

    # ---------------------------------------------------------------- 2
    rule("2. model")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels)).to(device)
    dtypes = {p.dtype for p in model.parameters()}
    print(f"  dtypes       {dtypes}")
    print(f"  num_labels   {model.config.num_labels}")
    print(f"  vocab (cfg)  {model.config.vocab_size}")
    if max(enc["input_ids"]) >= model.config.vocab_size:
        print("  !! token id exceeds model vocab - tokenizer/model mismatch")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters   {n_params / 1e6:.0f}M")

    # ---------------------------------------------------------------- 3
    rule("3. one forward pass before any training")
    texts = [r["text"] for r in rows[:BATCH]]
    y = torch.tensor([idx[r["label"]] for r in rows[:BATCH]]).to(device)
    batch = tok(texts, truncation=True, max_length=MAX_LEN,
                padding="max_length", return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        out = model(**batch, labels=y)
    print(f"  loss         {out.loss.item():.4f}   "
          f"(ln(77) = {np.log(77):.4f} is the untrained expectation)")
    print(f"  logits       finite={torch.isfinite(out.logits).all().item()}  "
          f"std={out.logits.std().item():.4f}")
    if not torch.isfinite(out.loss):
        print("  !! loss is not finite before training even begins")

    # ---------------------------------------------------------------- 4
    rule(f"4. {STEPS} training steps at three learning rates")
    print(f"  {'lr':>8s}  {'step 1':>8s} {'step 10':>8s} {'step 20':>8s} "
          f"{'step 40':>8s}  {'grad norm':>10s}  {'moved?':>7s}")

    for lr in LRS:
        torch.manual_seed(1)
        m = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=len(labels)).to(device)
        before = m.classifier.weight.detach().clone()
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        m.train()

        losses, gnorm = [], 0.0
        order = np.random.RandomState(0).permutation(len(rows))
        for step in range(STEPS):
            sel = [rows[order[(step * BATCH + j) % len(rows)]]
                   for j in range(BATCH)]
            b = tok([r["text"] for r in sel], truncation=True,
                    max_length=MAX_LEN, padding="max_length",
                    return_tensors="pt").to(device)
            yy = torch.tensor([idx[r["label"]] for r in sel]).to(device)
            opt.zero_grad()
            loss = m(**b, labels=yy).loss
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0).item()
            opt.step()
            losses.append(loss.item())

        moved = (m.classifier.weight.detach() - before).abs().max().item()
        picks = [losses[0], losses[9], losses[19], losses[-1]]
        cells = "  ".join(f"{v:8.4f}" if np.isfinite(v) else f"{'NaN':>8s}"
                          for v in picks)
        print(f"  {lr:8.0e}  {cells}  {gnorm:10.3f}  {moved:7.4f}")
        del m
        torch.cuda.empty_cache()

    print("\n  Loss should start near 4.34 and fall. Flat loss means nothing")
    print("  is being learned; NaN means overflow; 'moved' near zero means")
    print("  the optimiser is not updating the classifier head.")


if __name__ == "__main__":
    main()
