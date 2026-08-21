# Stage 3 — Inference

Built and validated before any baseline was run. Both paths here are
optimisations of something obvious and slow, and both fail *silently* when
wrong: left-padding a decoder incorrectly yields fluent nonsense, and a
mishandled KV cache yields plausible log-probabilities. Neither raises.

## Free-form: batched generation

Left padding, greedy, `max_new_tokens=16`. Validated against the unbatched
path item for item.

## Constrained: shared-prefix scoring

The prompt is encoded once; the first label token's log-probability comes
from the prompt's final logits, and the remaining tokens from a single
forward over all 77 label continuations against the expanded cache.

Cache-expansion APIs have moved across transformers versions, so three
strategies are attempted in order. On Kaggle (transformers 5.0.0) the
working one is `batch_repeat_interleave`. Measured speedup over the naive
path: **13.4–13.6×**.

Both scoring paths compute log-probabilities as `gather − logsumexp` rather
than `log_softmax(...).gather(...)`. The latter allocates a second
`[B, T, 151936]` tensor; at batch 16 over 470 positions that is a 4.5 GB
allocation for a dozen useful numbers, and it OOM'd a 14.56 GB card. Where
transformers supports `logits_to_keep`, only the final `longest + 1`
positions are computed at all.

## Batch size is a measurement parameter

Batched and unbatched greedy decoding **do not always agree**, and the
cached and naive scoring paths differ by 0.1–0.3 in log-probability.

The decisive test is not cached-vs-naive but naive-vs-naive:

| Comparison | max abs delta |
|---|---:|
| naive @4 vs naive @8 | 0.094 |
| naive @4 vs naive @16 | 0.094 |
| naive @4 vs naive @8 (item 1) | 0.045 |
| cached vs naive | 0.109 – 0.186 |

The reference disagrees with itself by the same magnitude that the fast
path disagrees with the reference. Changing batch size changes matmul
shapes and therefore fp16 reduction order. There is no bug to find; there
is a noise source to record.

Measured first-token logit noise from batching: **0.04 – 0.17**. Any item
whose top-two margin is smaller than that is decided by floating-point
reduction order rather than by the model.

**Consequences, fixed now:**

1. **Batch size is pinned and recorded in every run manifest**, exactly
   like a random seed. A result compared across two batch sizes is not a
   comparison.
2. **The noise floor has a third component.** Alongside decoding noise and
   training-seed variance, there is batching noise. It is cheap to measure
   — the same configuration at two batch sizes, two passes — and stage 5
   will do so rather than assume it is negligible.

### Where the noise lands

It does not spread evenly. Both anomalies observed during validation
occurred on `card_arrival` items, where several labels are genuinely
confusable; on a shuffled sample, at-risk items dropped to 0/8 and argmax
agreement became total.

Numerical noise flips predictions **precisely on the items that are
semantically ambiguous** — the same confusable clusters the stage-1 probe
identified. The two effects compound rather than being independent, which
matters when interpreting per-class error rates later.

Generation divergence also occurs mid-label rather than at the first token:
`top_up_failed` vs `top_up_reverted` share the prefix `top_up_`, so the
first sampled token is identical and the split happens at token three.
Banking77 labels cluster on shared prefixes, so this is the normal case,
not an oddity.

## `test.jsonl` is ordered by class

The upstream CSV groups rows by category and the frozen split preserves
that order. `test[:64]` is 64 items of a single intent.

Nothing already built is affected — full passes cover everything — but
**any subsample must be shuffled with a recorded seed**, or every timing
sample, every spot check and every debugging session is one class deep. The
first version of this check timed generation on 64 consecutive
`card_arrival` items and produced a sample output block where every gold
label was identical.

## Throughput and the budget

Measured, batch 32, Tesla T4, Qwen3-1.7B at fp16:

| | per item | full test pass (3,080) |
|---|---:|---:|
| free-form generation | 133 ms | 6.8 min |
| constrained, 77 labels | 581 ms | 29.8 min |

Peak VRAM 8.69 GB of 14.56 available.

Projected for the whole project, using the stage-3a token budgets:

| Work | Estimate |
|---|---:|
| Prompting baselines (zero-shot, few-shot ×3, retrieval) | ~6.5 h |
| LoRA training, 8 configs × 3 seeds | ~6 h |
| LoRA evaluation (bare-query prompts, 58 tokens) | ~5 h |
| **Total** | **~17 h** |

That fits inside one week of Kaggle's 30-hour quota with margin for the
reruns this project has already shown it needs.

### A held optimisation

For zero-shot and few-shot, the prompt prefix — instructions, the 77 label
names, and any fixed exemplars — is **byte-identical across all 3,080
items**. It could be encoded once for the entire pass instead of once per
item, cutting prefill by roughly 90% and bringing the k=77 few-shot
baseline from ~2.9 h to well under an hour.

Not built, because the budget above fits without it. Recorded here so that
if the grid tightens, the next place to find time is known rather than
rediscovered. It does not apply to retrieval, whose exemplars differ per
query.

## Artefacts

```
src/inference.py               both paths, plus the naive references
src/03a_check_inference.py     the validation run that produced these numbers
```

Validation functions kept in the codebase rather than deleted after use:
`verify_generation_equivalence`, `verify_scoring_equivalence`,
`scoring_self_consistency`, `diagnose_padding`. They are cheap, and any
future change to model, dtype or transformers version needs them again.
