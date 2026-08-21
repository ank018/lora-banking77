"""
Stage 3b check - prove both inference paths before running any baseline.

Nothing here is a result. It answers four questions whose wrong answers
are all silent:

  1. Does batched generation equal unbatched generation? (left padding)
  2. Does cached label scoring equal naive label scoring? (KV cache)
  3. Which cache-expansion strategy works on this transformers version?
  4. How long does a full pass over 3,080 items actually take?

Question 4 decides whether the ablation grid fits in Kaggle's weekly quota,
and it is answered by measurement rather than by my arithmetic.

    !python src/03a_check_inference.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import prompts as P  # noqa: E402
from inference import (  # noqa: E402
    batched_generate, cache_strategy, diagnose_padding, load_model, render,
    score_labels_cached, scoring_self_consistency,
    verify_generation_equivalence, verify_scoring_equivalence,
)

SUBSAMPLE_SEED = 20260821

N_GEN_CHECK = 8
N_TIMING = 64


def rule(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


def main():
    labels = P.load_labels()
    pool = P.load_pool()
    with open("eval/splits/test.jsonl", encoding="utf-8") as f:
        test_all = [json.loads(line) for line in f]

    # test.jsonl inherits upstream ordering, which is grouped by class:
    # test_all[:64] is a single intent. Any subsample must be shuffled or
    # every timing and every sample output is one class deep.
    import random
    test = list(test_all)
    random.Random(SUBSAMPLE_SEED).shuffle(test)
    print(f"labels {len(labels)}   test {len(test_all):,d}   "
          f"pool {len(pool):,d}")
    print(f"  first 8 gold labels unshuffled: "
          f"{len({r['label'] for r in test_all[:8]})} distinct")
    print(f"  first 8 gold labels shuffled:   "
          f"{len({r['label'] for r in test[:8]})} distinct "
          f"(seed {SUBSAMPLE_SEED})")

    model, tok = load_model()
    print(f"loaded {model.__class__.__name__} on {model.device}, "
          f"dtype {next(model.parameters()).dtype}")

    zs = [render(tok, P.zero_shot(r["text"], labels)) for r in test[:N_TIMING]]

    # ---------------------------------------------------------------- 1
    rule("1. batched vs unbatched generation (left padding)")
    g = verify_generation_equivalence(model, tok, zs[:N_GEN_CHECK])
    print(f"  {g['n']} prompts, {g['mismatches']} mismatches")
    for i, a, b in g["examples"]:
        print(f"    [{i}] batched={a.strip()[:40]!r}  single={b.strip()[:40]!r}")
    if g["mismatches"]:
        print("  !! batching is not equivalent - diagnosing")

    print("\n  first-token logits, batched vs unbatched")
    print(f"    {'i':>2s} {'pads':>5s} {'max|dlogit|':>12s} {'top2 gap':>9s} "
          f"{'flip':>5s}  verdict")
    for r in diagnose_padding(model, tok, zs[:N_GEN_CHECK]):
        if not r["flipped"]:
            verdict = "-"
        elif r["top2_gap"] < r["max_logit_delta"]:
            verdict = "near-tie, arithmetic"
        else:
            verdict = "REAL BUG: gap exceeds noise"
        print(f"    {r['i']:2d} {r['pad_tokens']:5d} "
              f"{r['max_logit_delta']:12.4f} {r['top2_gap']:9.4f} "
              f"{str(r['flipped']):>5s}  {verdict}")
    risky = [r for r in diagnose_padding(model, tok, zs[:N_GEN_CHECK])
             if r["at_risk"]]
    print(f"\n    at risk (top-2 margin inside batching noise): "
          f"{len(risky)}/{N_GEN_CHECK}")
    print("    Those items are decided by fp16 reduction order, not by the")
    print("    model. Batch size is a pinned, recorded parameter from here.")

    # ---------------------------------------------------------------- 2
    rule("2. cache-expansion strategy")
    strat = cache_strategy(model, tok, zs[0], labels)
    print(f"  strategy: {strat or 'NONE - falling back to the naive path'}")

    # ---------------------------------------------------------------- 3
    rule("3. cached vs naive label scoring")
    print("  first: is the reference self-consistent across batch sizes?")
    for i in range(2):
        s = scoring_self_consistency(model, tok, zs[i], labels)
        deltas = {k: v for k, v in s.items() if k.startswith("max_delta")}
        print(f"    item {i}  " + "  ".join(f"{k.split('_at_')[-1]}:{v:.4f}"
                                            for k, v in deltas.items())
              + f"   argmax_stable={s['argmax_stable']}")
    print("    If these deltas match the cached-vs-naive deltas below, the")
    print("    difference is fp16 arithmetic and there is no bug to fix.\n")

    for i in range(3):
        v = verify_scoring_equivalence(model, tok, zs[i], labels)
        print(f"  item {i}  max|delta| {v['max_abs_delta']:.4f}  "
              f"mean {v['mean_abs_delta']:.4f}  "
              f"within_tol {v['within_tolerance']}  "
              f"argmax_agrees {v['argmax_agrees']}")
        print(f"           naive {v['naive_s']:.2f}s -> cached "
              f"{v['cached_s']:.2f}s  ({v['speedup']:.1f}x)  "
              f"top1={v['top1_cached']}  gold={test[i]['label']}")

    # ---------------------------------------------------------------- 4
    rule("4. throughput, and what a full pass costs")
    torch.cuda.synchronize()
    t0 = time.time()
    outs = batched_generate(model, tok, zs, batch_size=32, max_new_tokens=16,
                            progress=False)
    torch.cuda.synchronize()
    gen_s = time.time() - t0
    per = gen_s / len(zs)
    print(f"  free-form  {len(zs)} items in {gen_s:.1f}s "
          f"({per * 1000:.0f} ms/item, batch 32)")
    print(f"             full test pass: {per * len(test) / 60:.1f} min")

    t0 = time.time()
    for p in zs[:16]:
        score_labels_cached(model, tok, p, labels)
    torch.cuda.synchronize()
    sc_s = (time.time() - t0) / 16
    print(f"  constrained {sc_s * 1000:.0f} ms/item (77 labels, cached)")
    print(f"             full test pass: {sc_s * len(test) / 60:.1f} min")

    rule("sample free-form outputs")
    for r, o in list(zip(test, outs))[:8]:
        print(f"  gold {r['label']:35s} -> {o.strip()[:50]!r}")

    print("\n  peak vram "
          f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    main()
