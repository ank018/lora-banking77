"""
Model fault or pipeline fault?

Established so far:
  - fp16 destroyed DeBERTa; fp32 fixed the NaNs but not the training
  - all three encoders memorise 32 examples perfectly (loss -> 0.0003)
  - the LR scheduler is not to blame: no-scheduler runs fail too, and the
    learning rate is exactly what it should be
  - on real rung-8 data, loss does not move in 200 steps at any setting

A caveat on the overfit test worth stating plainly: memorising 32 examples
proves gradients flow. It does NOT prove the labels are correct - a model
will memorise arbitrary text->label pairs just as happily. So a data
problem is still live.

Two hypotheses, one experiment:

  A  deberta-v3 fails at this scale on this stack
     -> roberta-base learns on identical data through identical code
  B  something in our pipeline is wrong
     -> roberta-base fails too, and the fault is ours

Long enough to be conclusive: 1,000 steps at a learning rate already shown
to work, no scheduler, real rung-8 data. Both train and dev accuracy are
reported - train rising while dev stays flat would mean memorisation
without generalisation, which points at the labels.

    python src/05e_model_vs_pipeline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

SPLITS = Path("eval/splits")
MODELS = ["roberta-base", "microsoft/deberta-v3-base"]
LR = 5e-5
BATCH = 16
STEPS = 1000
MAX_LEN = 64
REPORT_EVERY = 100


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def accuracy(model, enc, y, device):
    model.eval()
    preds = []
    for i in range(0, len(y), 64):
        b = {"input_ids": enc["input_ids"][i:i + 64].to(device),
             "attention_mask": enc["attention_mask"][i:i + 64].to(device)}
        preds.append(model(**b).logits.argmax(-1).cpu())
    model.train()
    return (torch.cat(preds) == y).float().mean().item()


def run(model_id, train_rows, dev_rows, labels, device):
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(labels), dtype=torch.float32).to(device)

    idx = {lbl: i for i, lbl in enumerate(labels)}
    enc_tr = tok([r["text"] for r in train_rows], truncation=True,
                 max_length=MAX_LEN, padding="max_length",
                 return_tensors="pt")
    y_tr = torch.tensor([idx[r["label"]] for r in train_rows])
    enc_dv = tok([r["text"] for r in dev_rows], truncation=True,
                 max_length=MAX_LEN, padding="max_length",
                 return_tensors="pt")
    y_dv = torch.tensor([idx[r["label"]] for r in dev_rows])

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    rng = np.random.RandomState(0)
    model.train()

    print(f"\n  {model_id}")
    print(f"    {'step':>6s} {'loss':>8s} {'train acc':>10s} {'dev acc':>9s}")
    window = []
    for step in range(1, STEPS + 1):
        sel = rng.choice(len(train_rows), BATCH, replace=False)
        b = {"input_ids": enc_tr["input_ids"][sel].to(device),
             "attention_mask": enc_tr["attention_mask"][sel].to(device)}
        opt.zero_grad()
        loss = model(**b, labels=y_tr[sel].to(device)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        window.append(loss.item())

        if step % REPORT_EVERY == 0 or step == 1:
            tr = accuracy(model, enc_tr, y_tr, device)
            dv = accuracy(model, enc_dv, y_dv, device)
            print(f"    {step:6d} {np.mean(window):8.4f} "
                  f"{tr * 100:9.1f}% {dv * 100:8.1f}%", flush=True)
            window = []

    del model
    torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = {r["id"]: r for r in read_jsonl(SPLITS / "train_pool.jsonl")}
    labels = sorted({r["label"] for r in pool.values()})
    ids = json.loads((SPLITS / "rungs" / "rung_08.json").read_text())
    train_rows = [pool[i] for i in ids]
    dev_rows = read_jsonl(SPLITS / "dev.jsonl")

    print(f"train {len(train_rows)} ({len({r['label'] for r in train_rows})} "
          f"labels)   dev {len(dev_rows)}   lr {LR}   "
          f"batch {BATCH}   {STEPS} steps, no scheduler")
    print(f"chance = {100 / len(labels):.1f}%")

    for model_id in MODELS:
        run(model_id, train_rows, dev_rows, labels, device)

    print("\n  roberta learns, deberta does not  -> model fault, swap encoder")
    print("  neither learns                     -> pipeline fault, ours")
    print("  train climbs, dev flat             -> labels are wrong")


if __name__ == "__main__":
    main()
