"""
Stage 5 - LoRA fine-tuning.

The thing this project was nominally about. It arrives with the bar
already set by stage 4: roberta-base scores 94.0% +/- 0.23 on the full
pool, from 125M parameters and 30 minutes on a free T4. Qwen3-1.7B is
2.03B parameters, 16x larger, and costs orders of magnitude more per
prediction. For fine-tuning it to be the right answer here, it has to
clear 94.0% by a margin that survives a paired test.

Design decisions, fixed before the first run:

  prompt        bare query, no label list (58 tokens vs zero-shot's 447).
                After training the label space is in the weights. This
                gives the fine-tuned model strictly LESS prompt context
                than every baseline it is compared against, which is the
                conservative direction for the handicap to run.

  centre config r=16, alpha=32, dropout 0.05, lr 2e-4, targeting the four
                attention projections plus the MLP triple. Conventional
                defaults chosen before any run, so stage 6's sweeps are
                honest variations rather than a search presented as an
                ablation.

  loss          computed on the label tokens only. Prompt positions are
                masked to -100; training the model to reproduce its own
                instructions would waste capacity on text it never has to
                generate.

  dtype         fp16. The T4 is sm75 and has no native bf16 - see
                docs/00_environment.md, where torch.cuda.is_bf16_supported()
                claims otherwise.

  selection     best epoch chosen on dev by free-form accuracy, never on
                test. Free-form rather than constrained because dev runs
                every epoch and generation is roughly 5x cheaper here.

    python src/04_train_lora.py --rungs 8 --seeds 1        # smoke
    python src/04_train_lora.py                             # sweep
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

import prompts as P  # noqa: E402
from evaluator import evaluate_constrained, evaluate_free_form, score  # noqa: E402
from runner import write_env  # noqa: E402

hf_logging.set_verbosity_error()

SPLITS = Path("eval/splits")
RUNS = Path("reports/runs")
ADAPTERS = Path("artifacts/adapters")
MODEL = "Qwen/Qwen3-1.7B"

RANK = 16
ALPHA = 32
DROPOUT = 0.05
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]
LR = 2e-4
EPOCHS = 8
BATCH = 8
GRAD_ACCUM = 2          # effective batch 16, matching the encoder
MAX_LEN = 96            # 58 prompt + label + slack; p95 query is 24 words
EVAL_BATCH = 32
SEEDS = [1, 2, 3]
RUNGS = [2, 4, 8, 16, 24]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_ids():
    m = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    thr = m["near_dup_threshold_recorded"]
    return [r["id"] for r in read_jsonl(SPLITS / "near_dup_test.jsonl")
            if r["twin_sim"] < thr]


def load_rung(k, pool_by_id):
    if k == 0:
        return list(pool_by_id.values())
    ids = json.loads((SPLITS / "rungs" / f"rung_{k:02d}.json")
                     .read_text(encoding="utf-8"))
    return [pool_by_id[i] for i in ids]


# ------------------------------------------------------------- encoding
def encode_example(tok, query, label):
    """prompt + label + eos, with loss masked to the label tokens only."""
    from inference import render
    prompt = render(tok, P.bare(query))
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    l_ids = tok(label, add_special_tokens=False).input_ids + [tok.eos_token_id]
    ids = (p_ids + l_ids)[:MAX_LEN]
    labels = ([-100] * len(p_ids) + l_ids)[:MAX_LEN]
    return ids, labels


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, mask = [], [], []
    for ids, lab in batch:
        gap = width - len(ids)
        input_ids.append(ids + [pad_id] * gap)
        labels.append(lab + [-100] * gap)
        mask.append([1] * len(ids) + [0] * gap)
    return (torch.tensor(input_ids), torch.tensor(labels),
            torch.tensor(mask))


# ------------------------------------------------------------- training
def build_model(tok):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda")
    model.config.use_cache = False
    cfg = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT,
                     target_modules=TARGETS, task_type="CAUSAL_LM",
                     bias="none")
    model = get_peft_model(model, cfg)

    # LoRA parameters stay in fp32. Adam moments on fp16 parameters
    # underflow, and on Turing there is no bf16 to fall back to.
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


@torch.no_grad()
def free_form_accuracy(model, tok, rows, labels):
    from inference import batched_generate, render
    model.config.use_cache = True
    model.eval()
    texts = [render(tok, P.bare(r["text"])) for r in rows]
    raws = batched_generate(model, tok, texts, batch_size=EVAL_BATCH,
                            max_new_tokens=16, progress=False)
    model.config.use_cache = False
    model.train()
    ok = sum(1 for r, raw in zip(rows, raws)
             if evaluate_free_form(raw, r["label"], labels,
                                   exclude=r["text"])["verdict"] == "correct")
    return ok / len(rows)


def train_one(train_rows, dev_rows, labels, seed, tok, epochs):
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(tok)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    encoded = [encode_example(tok, r["text"], r["label"]) for r in train_rows]
    steps_per_epoch = max(1, len(encoded) // (BATCH * GRAD_ACCUM))
    total = steps_per_epoch * epochs

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)

    rng = np.random.RandomState(seed)
    best = {"acc": -1.0, "epoch": -1, "state": None}
    history = []

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(encoded))
        losses = []
        for step in range(steps_per_epoch):
            opt.zero_grad()
            for micro in range(GRAD_ACCUM):
                lo = (step * GRAD_ACCUM + micro) * BATCH
                sel = order[lo:lo + BATCH]
                if len(sel) == 0:
                    continue
                ids, lab, mask = collate([encoded[i] for i in sel],
                                         tok.pad_token_id)
                out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(),
                            labels=lab.cuda())
                (out.loss / GRAD_ACCUM).backward()
                losses.append(out.loss.item())
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()

        mean_loss = float(np.mean(losses))
        if not np.isfinite(mean_loss):
            raise RuntimeError(f"loss is {mean_loss} at epoch {epoch}")
        acc = free_form_accuracy(model, tok, dev_rows, labels)
        history.append({"epoch": epoch, "loss": round(mean_loss, 4),
                        "dev_acc": round(acc, 4)})
        print(f"      ep{epoch:<2d} loss {mean_loss:6.3f}  dev {acc * 100:5.1f}%",
              flush=True)
        if acc > best["acc"]:
            best = {"acc": acc, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()
                              if "lora" in k.lower()}}

    model.load_state_dict(best["state"], strict=False)
    if best["acc"] <= 2.0 / len(labels):
        print(f"      !! dev accuracy {best['acc'] * 100:.1f}% is at chance")
    return model, {"dev_acc": best["acc"], "best_epoch": best["epoch"],
                   "history": history, "trainable": n_trainable,
                   "total_params": n_total}


# ----------------------------------------------------------------- eval
def evaluate_test(model, tok, test, labels, regimes):
    from inference import (batched_generate, encode_labels,
                           plan_generation_batch, plan_label_chunk,
                           render, score_labels_cached)
    model.config.use_cache = True
    model.eval()
    texts = [render(tok, P.bare(r["text"])) for r in test]
    n_tok = max(len(tok(t).input_ids) for t in texts[:64])
    out = {}

    if "free_form" in regimes:
        gb = min(EVAL_BATCH, plan_generation_batch(n_tok))
        raws = batched_generate(model, tok, texts, batch_size=gb,
                                max_new_tokens=16, progress=False)
        out["free_form"] = [
            evaluate_free_form(raw, r["label"], labels, exclude=r["text"])
            | {"id": r["id"], "query": r["text"]}
            for r, raw in zip(test, raws)]

    if "constrained" in regimes:
        chunk = plan_label_chunk(n_tok, len(labels))
        label_ids = encode_labels(tok, labels)
        recs = []
        t0 = time.time()
        for i, r in enumerate(test):
            sc = score_labels_cached(model, tok, texts[i], labels,
                                     label_ids, chunk_size=chunk)
            order = sorted(range(len(labels)), key=lambda j: -sc[j])
            rec = evaluate_constrained(labels[order[0]], r["label"], labels)
            rec |= {"id": r["id"], "query": r["text"],
                    "top2_margin": round(sc[order[0]] - sc[order[1]], 4),
                    "top5": [[labels[j], round(sc[j], 4)] for j in order[:5]]}
            recs.append(rec)
            if (i + 1) % 512 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"      constrained {i + 1}/{len(test)}  "
                      f"{rate:.1f}/s", flush=True)
                torch.cuda.empty_cache()
        out["constrained"] = recs

    out["_prompt_tokens"] = n_tok
    return out


def write_run(runs, name, regime, recs, meta):
    d = runs / f"{name}__{regime}"
    d.mkdir(parents=True, exist_ok=True)
    write_env(d)
    with open(d / "predictions.jsonl", "w", encoding="utf-8",
              newline="\n") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    (d / "meta.json").write_text(json.dumps(meta, indent=2),
                                 encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--rungs", nargs="*", type=int, default=RUNGS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--regimes", nargs="*",
                    default=["free_form", "constrained"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--save-adapter", action="store_true")
    # Two configurations that differ in anything must not share a run
    # directory. Without a tag, an 8-epoch rerun overwrites the committed
    # 4-epoch results and the comparison loses the thing it compares to.
    ap.add_argument("--tag", default="",
                    help="suffix for run names, e.g. ep8")
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    pool = read_jsonl(SPLITS / "train_pool.jsonl")
    pool_by_id = {r["id"]: r for r in pool}
    dev = read_jsonl(SPLITS / "dev.jsonl")
    test = read_jsonl(SPLITS / "test.jsonl")
    labels = sorted({r["label"] for r in pool})
    clean = clean_ids()

    print(f"{MODEL}  r={RANK} alpha={ALPHA} lr={LR} epochs={args.epochs}")
    print(f"rungs {args.rungs}  seeds {args.seeds}  "
          f"test {len(test):,d}  dev {len(dev):,d}")
    print(f"bar to clear: robertabase full pool 94.0% +/- 0.23 pp")

    for k in args.rungs:
        rung = load_rung(k, pool_by_id)
        for seed in args.seeds:
            label_k = "full" if k == 0 else f"{k:02d}"
            suffix = f"_{args.tag}" if args.tag else ""
            name = f"qwen3lora_rung{label_k}{suffix}_seed{seed}"
            done = all((runs / f"{name}__{rg}" / "predictions.jsonl").exists()
                       for rg in args.regimes)
            if done:
                print(f"  {name} already complete")
                continue

            print(f"\n  {name}  n_train={len(rung):,d}", flush=True)
            t0 = time.time()
            model, info = train_one(rung, dev, labels, seed, tok, args.epochs)
            train_s = time.time() - t0
            print(f"      trained in {train_s / 60:.1f} min   "
                  f"{info['trainable']:,d} trainable of "
                  f"{info['total_params']:,d} "
                  f"({info['trainable'] / info['total_params'] * 100:.2f}%)")

            preds = evaluate_test(model, tok, test, labels, args.regimes)
            meta = {
                "config": name, "model": MODEL, "rung_per_class": k,
                "n_train": len(rung), "seed": seed,
                "rank": RANK, "alpha": ALPHA, "dropout": DROPOUT,
                "targets": TARGETS, "lr": LR, "epochs": args.epochs,
                "batch": BATCH, "grad_accum": GRAD_ACCUM,
                "max_len": MAX_LEN, "eval_batch": EVAL_BATCH,
                "prompt_versions": P.VERSIONS,
                "prompt_tokens": preds["_prompt_tokens"],
                "best_epoch": info["best_epoch"], "dev_acc": info["dev_acc"],
                "history": info["history"],
                "trainable_params": info["trainable"],
                "train_seconds": round(train_s), "n_items": len(test),
            }
            for rg in args.regimes:
                recs = preds[rg]
                write_run(runs, name, rg, recs, meta | {"regime": rg})
                s, c = score(recs), score(recs, subset_ids=clean)
                print(f"      {rg:12s} test {s['accuracy'] * 100:5.1f}%  "
                      f"clean {c['accuracy'] * 100:5.1f}%  "
                      f"unparseable {s['unparseable_rate'] * 100:4.1f}%")

            if args.save_adapter:
                model.save_pretrained(ADAPTERS / name)
            del model
            torch.cuda.empty_cache()

    print("\n  Aggregate with src/summarise_runs.py --filter qwen3lora")
    print("  Compare with src/compare.py against robertabase_rungfull.")


if __name__ == "__main__":
    main()
