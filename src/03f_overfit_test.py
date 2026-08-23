"""
Can the training loop learn anything at all?

After fixing the fp16 overflow, DeBERTa-v3 still does not train: 20 epochs,
780 gradient steps, loss 4.361 -> 4.304, dev accuracy 1.3% -> 2.8%. A 184M
model given 616 examples should be memorising them outright by then.

Two possibilities, and the previous diagnostic cannot separate them:

  the loop is broken     labels, loss, or optimiser wired wrong, in which
                         case no model will learn
  the model is broken    deberta-v3 specifically fails in this
                         transformers version, in which case another
                         encoder trains fine on the identical loop

The test is the standard one: overfit 32 examples. Any correctly wired
classifier drives loss to near zero and train accuracy to 100% within a
few hundred steps on 32 examples - it has enough capacity to memorise them
and no reason not to. A model that cannot is misconfigured.

Running the same loop over three encoders makes the answer unambiguous.

    python src/03f_overfit_test.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

SPLITS = Path("eval/splits")
MODELS = ["microsoft/deberta-v3-base", "roberta-base", "bert-base-uncased"]
N = 32
STEPS = 300
LR = 5e-5
MAX_LEN = 64
CHECKS = [1, 50, 100, 200, 300]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def overfit(model_id, rows, labels, device):
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(labels), dtype=torch.float32).to(device)

    idx = {lbl: i for i, lbl in enumerate(labels)}
    batch = tok([r["text"] for r in rows], truncation=True,
                max_length=MAX_LEN, padding="max_length",
                return_tensors="pt").to(device)
    y = torch.tensor([idx[r["label"]] for r in rows]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    trace = {}
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        out = model(**batch, labels=y)
        out.loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        if step in CHECKS:
            trace[step] = (out.loss.item(), gn)

    model.eval()
    with torch.no_grad():
        pred = model(**batch).logits.argmax(-1)
    acc = (pred == y).float().mean().item()
    n_distinct = len(set(pred.tolist()))
    del model
    torch.cuda.empty_cache()
    return trace, acc, n_distinct


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = {r["id"]: r for r in read_jsonl(SPLITS / "train_pool.jsonl")}
    labels = sorted({r["label"] for r in pool.values()})
    ids = json.loads((SPLITS / "rungs" / "rung_08.json").read_text())
    rows = [pool[i] for i in ids[:N]]
    print(f"overfitting {N} examples, {len(set(r['label'] for r in rows))} "
          f"distinct labels, {STEPS} steps, lr {LR}, {device}")

    print(f"\n  {'model':26s} " + " ".join(f"{'s' + str(c):>8s}"
                                           for c in CHECKS)
          + f" {'train acc':>10s} {'distinct':>9s}")
    for model_id in MODELS:
        try:
            trace, acc, nd = overfit(model_id, rows, labels, device)
            cells = " ".join(f"{trace[c][0]:8.4f}" for c in CHECKS)
            flag = "" if acc > 0.9 else "   <- cannot memorise 32 examples"
            print(f"  {model_id:26s} {cells} {acc * 100:9.1f}% "
                  f"{nd:8d}{flag}")
        except Exception as e:  # noqa: BLE001
            print(f"  {model_id:26s} FAILED {type(e).__name__}: {e}")

    print(f"\n  Any correctly wired classifier reaches ~100% train accuracy")
    print(f"  on {N} examples within {STEPS} steps. If all three fail, the")
    print("  loop is wrong. If only deberta-v3 fails, the model is at fault")
    print("  in this transformers version and we swap encoders.")


if __name__ == "__main__":
    main()
