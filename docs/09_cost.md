# Stage 9 — Cost and latency

Accuracy is one axis. Every table in this project up to here reported only
that one, and a method matching another at 110× the time per prediction is
not equivalent to it.

All figures are read from committed `meta.json` files — training seconds,
prompt tokens, batch and chunk sizes were recorded at run time — so the
table regenerates without a GPU and cannot drift from the runs it describes.

## The table

Tesla T4, fp16, single-item constrained scoring. GPU rate $0.35/hour
(RTX 4090 community tier, August 2026) — a parameter, not a measurement.

| Method | Accuracy | Train | Prompt | items/s | ms/item | $/1k |
|---|---:|---:|---:|---:|---:|---:|
| majority class | 1.3% | — | — | — | — | — |
| kNN over TF-IDF (CPU) | 82.9% | — | — | 3,000 | 0.3 | free |
| base model, bare prompt | 31.4% | — | 63 | 4.6 | 217 | $0.021 |
| zero-shot, 77 labels listed | 47.1% | — | 447 | 2.0 | 500 | $0.049 |
| few-shot, 77 exemplars | 5.7% | — | 2,127 | 0.4 | 2,500 | $0.243 |
| retrieval + LLM | 83.1% | — | 728 | 1.5 | 667 | $0.065 |
| **roberta-base, fine-tuned** | **94.0%** ±0.23 | 31 min | 64 | 330 | **3.0** | $0.0003 |
| **LoRA on Qwen3-1.7B** | **93.6%** ±0.26 | 76 min | 63 | 3.0 | **333** | $0.032 |
| LoRA, rank 64 | 93.5% | 94 min | 63 | 2.7 | 370 | $0.036 |

## The cost argument is smaller than I kept saying

Through several stages this project described LoRA as costing "roughly
100× more per prediction" as though that settled something. In relative
terms it does: 110× slower than the encoder. In dollars it does not.

At one million predictions a month:

| Method | GPU-hours | Cost |
|---|---:|---:|
| roberta-base | 0.8 | $0 |
| kNN | 0.1 | $0 (CPU) |
| LoRA Qwen3-1.7B | 92.6 | **$32** |
| few-shot, 77 exemplars | 694.4 | $243 |

**The entire spread is $0 to $243 a month** — less than one engineer-hour.
No method here is expensive, and a reader could check that in ten seconds
and rightly discount any argument built on it.

What the numbers do support:

**Latency.** 3 ms against 333 ms per prediction. That matters for an
interactive router where a customer is waiting, and not at all for
overnight batch triage.

**Deployment.** roberta-base and kNN run on CPU. The LoRA model needs a GPU
to be practical at 333 ms/item. That is an infrastructure decision — what
you have to provision and maintain — rather than a line item.

**Capacity.** 0.8 against 93 GPU-hours a month. Relevant if you own the
hardware; irrelevant if you rent by the hour.

**The honest headline remains what it was: the 16× larger model is not more
accurate** (93.59% ± 0.26 against 93.99% ± 0.23, p = 0.12). Cost is a
supporting argument about latency and deployability, not a knockout blow.

Where cost *does* decide something is the retrieval row below, and there it
is not close.

## The retrieval row: 2,200× for nothing

`retrieval + LLM` at 83.1% and 667 ms/item is the most expensive way in
this table to get an answer that was already available.

Stage 4 established that **kNN over the same retrieved neighbours scores
82.9%** — a difference of 0.2 points at p = 0.18, indistinguishable — and
that the language model's answer equals the top-1 retrieved neighbour's
label **81.4% of the time**. It is mostly repeating its own retriever.

| | accuracy | ms/item |
|---|---:|---:|
| kNN over TF-IDF | 82.9% | 0.3 |
| the same retrieval, then ask a 2B model | 83.1% | 667 |

**The language model costs 2,200× the retriever it is reading from, and the
paired test cannot distinguish their outputs.** Framed against roberta-base
the row reads as 220× slower; framed against the component actually doing
the work, it is an order of magnitude worse than that.

This is the cleanest cost-per-accuracy failure in the project, and it is
the architecture a great many production RAG-for-classification systems
use. The fair caveat from `docs/04_baselines.md` still applies: the LLM
does beat 1-NN (82.7 vs 80.2), so it is not purely copying. It just does
not beat counting the neighbours, which costs nothing.

## Cost and accuracy are independent axes

The clearest row in the table is `few-shot with 77 exemplars`: **825×
slower than roberta-base and 88 points less accurate.** The most expensive
method in the project is also its second worst.

Prompt length drives inference cost almost entirely — 2,127 tokens against
63 — and buys nothing, because in-context examples collapse the model's
predictions (`docs/04_baselines.md`). Anyone reaching for many-shot
prompting to avoid a training run would be paying eight times a fine-tuned
model's inference cost for a tenth of its accuracy.

The fine-tuned model's 63-token prompt is the other side of this. Dropping
the 77-label list saves 384 tokens per call, and the label space lives in
the weights instead.

## What these numbers are not

**Not a serving benchmark.** Single-item scoring on a T4 with 77 label
continuations per item. A production system would batch requests, quantise
weights, and cache the shared prompt prefix across items — all of which
help the LoRA model more than roberta-base, since it has more to amortise.
The 110× ratio is an upper bound on the gap, not an estimate of it.

`docs/03_inference.md` records a held optimisation of exactly this kind:
for zero-shot and few-shot the prompt prefix is byte-identical across all
3,080 items and could be encoded once per pass rather than once per item,
cutting prefill by ~90%. Not built, because the project's budget fitted
without it.

**Not a total cost of ownership.** No engineering time, no serving
infrastructure, no monitoring, no retraining cadence. The 76 minutes of
LoRA training against roberta's 31 is wall clock on a free Kaggle T4, and
includes per-epoch dev evaluation because that is what the runs did.

**Not a fixed rate.** `--gpu-rate` is a flag. Rates move, and the
conclusion — that the dollar spread is negligible at this volume — holds
across any plausible value.

## Where the crossover changes the answer

Under ~500 labelled examples, LoRA is worth **+8.93 pp** over the encoder
(`docs/06_lora.md`). At that accuracy level the latency and infrastructure
arguments do not outweigh nine points of accuracy, so the larger model is
the right call despite costing more.

Above ~500 examples the accuracy difference disappears and the cost
arguments are all that remain. That is where they decide the question — not
because they are large, but because nothing else separates the two.

| labelled examples | choose | why |
|---|---|---|
| under ~500 | LoRA on the LLM | +8.93 pp; cost arguments do not outweigh it |
| over ~500 | roberta-base | indistinguishable accuracy, 110× faster, CPU-deployable |

## Artefacts

```
src/08_cost_latency.py    reads meta.json; --gpu-rate and --volume are flags
```
