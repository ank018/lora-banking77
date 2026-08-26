"""
Why does the real run fail when the overfit test succeeds?

All three encoders memorise 32 examples perfectly (loss 4.32 -> 0.0003,
100% train accuracy in 300 steps), so the model and the loop are sound.
Yet the real rung-8 run sat at loss 4.30 after 780 steps.

The two setups differ in four ways. Loss reached 0.046 in 50 steps without
a scheduler and went nowhere in 780 steps with one, which a 2.5x learning
rate difference cannot account for. So the scheduler is the suspect, and
the fix depends on which factor is actually responsible.

Four configurations on the real rung-8 data, 200 steps each, identical in
every other respect. The learning rate is printed as the optimiser sees it,
because a scheduler that has silently zeroed it looks exactly like a model
that will not learn.

    python src/05d_isolate_training.py
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
MODEL = "microsoft/deberta-v3-base"
BATCH = 16
STEPS = 200
MAX_LEN = 64
WARMUP_RATIO = 0.1
CHECKS = [1, 25, 50, 100, 200]

CONFIGS = [
    ("no sched, lr 5e-5", 5e-5, False),
    ("no sched, lr 2e-5", 2e-5, False),
    ("warmup,   lr 5e-5", 5e-5, True),
    ("warmup,   lr 2e-5", 2e-5, True),   # what the failing run used
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run(rows, labels, lr, use_sched, device):
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels), dtype=torch.float32).to(device)

    idx = {lbl: i for i, lbl in enumerate(labels)}
    enc = tok([r["text"] for r in rows], truncation=True, max_length=MAX_LEN,
              padding="max_length", return_tensors="pt")
    y_all = torch.tensor([idx[r["label"]] for r in rows])

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = (get_linear_schedule_with_warmup(
        opt, int(STEPS * WARMUP_RATIO), STEPS) if use_sched else None)

    rng = np.random.RandomState(0)
    model.train()
    trace, lrs = {}, []
    for step in range(1, STEPS + 1):
        sel = rng.choice(len(rows), BATCH, replace=False)
        b = {"input_ids": enc["input_ids"][sel].to(device),
             "attention_mask": enc["attention_mask"][sel].to(device)}
        opt.zero_grad()
        loss = model(**b, labels=y_all[sel].to(device)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched:
            sched.step()
        lrs.append(opt.param_groups[0]["lr"])
        if step in CHECKS:
            trace[step] = loss.item()

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(rows), 64):
            b = {"input_ids": enc["input_ids"][i:i + 64].to(device),
                 "attention_mask": enc["attention_mask"][i:i + 64].to(device)}
            preds.append(model(**b).logits.argmax(-1).cpu())
        acc = (torch.cat(preds) == y_all).float().mean().item()

    del model
    torch.cuda.empty_cache()
    return trace, acc, lrs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = {r["id"]: r for r in read_jsonl(SPLITS / "train_pool.jsonl")}
    labels = sorted({r["label"] for r in pool.values()})
    ids = json.loads((SPLITS / "rungs" / "rung_08.json").read_text())
    rows = [pool[i] for i in ids]
    print(f"rung 8: {len(rows)} examples, "
          f"{len({r['label'] for r in rows})} labels, batch {BATCH}, "
          f"{STEPS} steps")

    print(f"\n  {'config':20s} " + " ".join(f"{'s' + str(c):>8s}"
                                            for c in CHECKS)
          + f" {'train acc':>10s} {'lr@1':>9s} {'lr@end':>9s}")
    for name, lr, use_sched in CONFIGS:
        trace, acc, lrs = run(rows, labels, lr, use_sched, device)
        cells = " ".join(f"{trace[c]:8.4f}" for c in CHECKS)
        print(f"  {name:20s} {cells} {acc * 100:9.1f}% "
              f"{lrs[0]:9.2e} {lrs[-1]:9.2e}")

    print("\n  Compare rows. If the two 'no sched' rows learn and the two")
    print("  'warmup' rows do not, the scheduler is the fault. If lr@1 is")
    print("  near zero under warmup, that is the mechanism. If all four")
    print("  behave the same, the difference lies in the data, not here.")


if __name__ == "__main__":
    main()
