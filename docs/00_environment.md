# Stage 0 — Environment and base model

Measured on Kaggle, 21 August 2026, before any training code existed. The
ordering is deliberate: the toolchain is the largest schedule risk in this
project, so it was stood up before the thing it would run.

## What Kaggle actually provides

| | |
|---|---|
| GPU | Tesla T4 × 2, compute capability **sm75** (Turing) |
| VRAM | **14.56 GB** usable per card, not the nominal 16 |
| CUDA | 12.8 |
| torch | 2.10.0+cu128 |
| transformers | **5.0.0** |
| peft | 0.19.1 |
| accelerate | 1.13.0 |
| trl, bitsandbytes | **absent** — must be installed at stage 4 |

Quota is ~30 GPU-hours/week with a ~12-hour session cap. Internet must be
switched on in the notebook sidebar, which requires phone verification on
the account.

## The bf16 trap

`torch.cuda.is_bf16_supported()` returned **`True`** on a Tesla T4.

It is not true in any sense that matters. Turing has no bf16 tensor cores;
native bfloat16 arrives with Ampere at compute capability 8.0. The
convenience API counts software emulation, so it reports `True` on hardware
that will run bf16 slowly through a compatibility path rather than refusing
it.

Nothing errors. A training run configured with `bf16=True` on this card
completes, takes far longer than it should, and offers no clue why. It
would also have been written into every run's `env.json` as a recorded
hardware fact, quietly poisoning any later comparison of timings.

Both the smoke test and `runner.py` now test **compute capability ≥ 8.0**
and record three values — `bf16_native`, `bf16_api`, and
`bf16_api_no_emulation` — so the discrepancy is preserved in the manifest
rather than resolved silently. The `including_emulation=False` keyword is
called defensively, since it does not exist on older torch builds.

**Every training configuration in this project uses fp16.** That was the
plan from the start; it is now the plan for a verified reason.

## Base model: Qwen3-1.7B

The decision rule was written into the smoke test *before* it ran:

> Take Qwen3.5-2B if it loads on Kaggle's transformers, exposes the four
> standard attention projections, and scores forced continuations.
> Otherwise take Qwen3-1.7B and record the reason.

**Qwen3.5-2B does not load.** Kaggle's transformers 5.0.0 does not
recognise model type `qwen3_5` — a hard `KeyError`, not a warning. It needs
a newer release or a git-main install.

Upgrading transformers in the notebook was considered and rejected. It is
an unpinnable dependency reinstalled every session; it puts PEFT 0.19.1 and
TRL against an architecture they may not target correctly; and the payoff
is a vision encoder carried for nothing on a nine-word text task. The rule
exists so that this is not relitigated after one option failed.

**Qwen3-1.7B, measured:**

| | |
|---|---|
| Class | `Qwen3ForCausalLM` |
| Parameters | **2.03 B** — the name counts non-embedding parameters; embeddings and `lm_head` are untied here, so both are materialised |
| Weights at fp16 | 3.79 GB, peak 3.84 GB |
| Layers | 28 |
| Linear targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` — all 28× |
| `enable_thinking=False` | accepted and honoured; no thinking block leaked |
| Constrained scoring | works — `logP("card_arrival") = −14.56` |

All four standard attention projections are present, so the rank ablation
uses the conventional target set with no asterisk.

### The first zero-shot output

Asked to classify *"I am still waiting on my card?"*, the untuned model
answered:

```
card_waiting
```

No such label exists in Banking77. The model invented a plausible one. Our
evaluator refuses it as `unparseable` — which is the free-form problem
demonstrated on the very first generation of the project, before any
baseline was run.

## Version divergence, on day one

Local Windows has transformers **5.15.1**; Kaggle has **5.0.0**. This is
exactly the split `env.json` was built to catch, and it appeared
immediately. Analysis runs locally against committed prediction files;
generation runs on Kaggle; the manifest records which environment produced
which artefact.

## Throughput, and what it forces

A single unbatched generation of 24 tokens took **1.07 s**.

Naively extrapolated, one free-form pass over the 3,080-item test set is
~50 minutes, and constrained scoring — 77 labels × 3,080 items as
independent forward passes — is hours per configuration. Across five
baselines, five rungs, a hyperparameter grid and three seeds, that exceeds
the weekly quota several times over.

So `inference.py` cannot be a loop. Two requirements, fixed now:

1. **Batch free-form generation.** Inputs are ~42 prompt tokens and outputs
   are short; batches of 32–64 should fit comfortably in 14.56 GB.
2. **Score constrained candidates against a shared prefix.** Compute the
   prompt's KV cache once per item and evaluate all 77 label continuations
   against it, rather than 77 independent forward passes.

This is real engineering and it is on the critical path. Discovering it
when the first ablation overran its budget would have cost a week.

## Artefacts

```
src/00b_smoke_gpu.py     the smoke test, rerunnable on any GPU host
src/runner.py            env_manifest() / write_env(), called by every stage
```
