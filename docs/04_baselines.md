# Stage 3 — Baselines

Six methods, one frozen 3,080-item test set, both decoding regimes. The
stage was planned as scaffolding for the fine-tuning experiment. It became
the most informative part of the project, and it moved the bar the
fine-tune has to clear from 45.8% to 82.9%.

## Results

| Method | Accuracy | 95% CI | Draws | Cost |
|---|---:|---|---:|---|
| majority class | 1.3% | ±0.4 | exact | — |
| few-shot, 77 examples | 5.7% | ±0.8 | **1** | GPU |
| few-shot, 20 examples | 3.9 / 4.6 / 12.0% | ±0.7–1.2 | 3 | GPU |
| few-shot, 5 examples | 22.0% | ±1.5 | **1** | GPU |
| zero-shot | 45.8% | ±1.8 | **1** | GPU |
| retrieval + LLM | 82.7% | ±1.3 | **1** | GPU |
| **kNN, k=5, no model** | **82.9%** | ±1.3 | exact | **CPU, seconds** |

CIs are unpaired binomial. Comparisons between rows use McNemar, which is
tighter because the two runs agree on most items and only the discordant
pairs carry information. **Bold "1" means unreplicated** — see Limitations.

## The language model is not earning its place

`retrieval_k10` prompts the model with the ten nearest training examples.
`knn_k5` does the retrieval and stops there — a similarity-weighted vote
over neighbours, no model, no GPU, a few seconds on a laptop.

Paired comparison on the same 3,080 items:

```
delta        +0.78 pp in kNN's favour (k=10; +0.2 for the dev-selected k=5)
95% CI       [-0.30, +1.86]
discordant   288 of 3,080 (9.4%)
McNemar p    0.18   -> not significant
```

The two are indistinguishable. At the very best the 2-billion-parameter
model is 0.3 points ahead.

The mechanism is direct: **the LLM's answer equals the top-1 retrieved
neighbour's label 81.4% of the time.** It is mostly repeating its retriever.

The fair version of the claim, which is more interesting than "kNN wins":
the LLM does beat 1-NN (82.7 vs 80.2), so it is not purely copying — when
it deviates it deviates usefully. But it does not beat a similarity-weighted
vote over the same neighbours. **The value the language model adds over its
retrieved context is worth about as much as counting them.**

### A methodological error, found and fixed

The first version of the kNN script ran k ∈ {1,3,5,10} on the **test set**
and reported the best (83.4%). That is selection on the test set — the
exact practice this project's rules forbid, committed while writing about
not committing it.

`k` is now chosen on dev (616 items), which selected **k=5**, giving 82.9%
on test. The inflation from best-of-k on test was **0.5 points**. All k
values are still printed, along with what best-of-k would have given, so
the selection is auditable rather than merely asserted.

| k | dev (616) | test (3,080) |
|---:|---:|---:|
| 1 | 82.3% | 80.2% |
| 3 | 83.4% | 81.9% |
| **5** | **84.7%** ← selected | **82.9%** ← reported |
| 10 | 83.3% | 83.4% |
| 20 | 83.1% | 81.7% |

**The reported figure is deliberately not the best test figure.** k=10
scores higher on test (83.4%), and dev's resolution at n=616 is roughly
±3 points, so k = 3, 5, 10 and 20 are not separable there — dev picked a
k it could not fully distinguish from its neighbours. That is the correct
behaviour, not a flaw. Noticing that k=10 does better on test and switching
would reintroduce exactly the error being corrected. The 0.5-point gap is
the honest price of not looking.

It surfaced from a question — "are we sure about these numbers?" — not from
a test. Worth remembering that the checks that catch this class of error
are social as often as automated.

Both figures are deterministic and reproduce exactly across machines; a
method with no model in it has no run-to-run variance to report.

## In-context examples destroy accuracy

Accuracy falls monotonically as exemplars are added: 45.8% with none, 22.0%
at k=5, 3.9–12.0% at k=20, 5.7% at k=77. At k≥20 the model barely beats the
1.3% majority floor.

**It is not an output-formatting artefact.** Constrained decoding — scoring
all 77 labels and taking the argmax, making malformed output impossible —
gives 5.7% at k=77 against free-form's 5.7%, disagreeing on **1 item in
3,080**. The model's beliefs have collapsed, not its formatting.

**The failure is mode collapse, not misclassification.** At k=77, two labels
account for 87% of all 3,080 predictions. At k=20 seed 2, one label takes
77.8%.

**Query sensitivity, measured directly.** Scoring all 77 labels for 40
queries spanning many classes and counting distinct argmax winners:

| Config | distinct labels over 40 queries |
|---|---:|
| zero-shot | 30 |
| few-shot k=20 (3 seeds) | 10, 5, 6 |
| few-shot k=77 | 8 |

The model stops responding to the customer's message.

### Two mechanisms proposed, both refuted

**Label-space collapse onto the exemplar set.** Predicted that the model
would answer only from the k demonstrated labels, capping accuracy at k/77.
Refuted: k=5 scored 22.0% against a 6.5% ceiling, and 53 distinct labels
were predicted.

**Recency anchoring.** The k=20 seed-1 run collapsed onto
`Refund_not_showing_up`, its own final exemplar, with 82% of
exemplar-matching predictions in the last decile of the list. Prediction:
all three seeds collapse onto their own last exemplar. Refuted, **1 of 3**.
Seed 2 collapsed onto `request_refund`, which *was never shown to it as an
exemplar at all*. Seed 1's apparent recency was coincidence — its last
exemplar happened to be a refund intent.

**Contentless-query prior.** Replacing the customer message with an empty
or generic string and scoring all labels matched the observed collapse
target in **2 of 4** configs. Partial support, not an explanation.

### What can be said

Exemplars push the model into a small attractor set of roughly five
high-prior intents — `request_refund`, `Refund_not_showing_up`,
`pending_transfer`, `pending_cash_withdrawal`,
`balance_not_updated_after_cheque_or_cash_deposit` — which recur as the
collapse target across every configuration and were already the most-predicted
labels at k=5. **Which member of that set wins is not predicted by either
mechanism tested.**

### Scope

One model (Qwen3-1.7B), one prompt layout, one task with a 77-way label
space. In-context learning normally helps, and published many-shot results
generally show gains. This may be a property of small models, of this
prompt format, or of large label spaces. **It is not evidence that
many-shot ICL fails in general**, and the write-up must not imply it.

## Limitations

- **Most rows are a single draw.** Only k=20 is replicated, and it showed an
  **8.2-point spread** across three exemplar draws. The k=5 and k=77 values
  are single draws with unknown bands; k=20 and k=77 are consequently
  **not separable**, and the "monotonic" shape beyond 20 exemplars is not
  established. The collapse from 0 to 20 is far too large to be draw noise.
- **Settings are not uniform.** `few_shot_k77` ran at generation batch 18
  where the rest used 32, because its prompts do not fit otherwise. Batch
  size is a measured noise source (docs/03_inference.md).
- **Reruns are not bit-identical.** A clean regeneration of
  `few_shot_k20_s2` differed from a resumed one by 2 rows in 3,080 —
  consistent with the batching noise measured in stage 3, appearing
  unprompted.
- **Zero-shot is unreplicated** and is the reference point for every
  few-shot comparison.
- The clean-subset column tracks full accuracy within ~1 point everywhere,
  as expected: no prompting method except retrieval can memorise, and
  retrieval's excess over the zero-shot calibration offset is ~0.6 points.

## Predictions, scored

Recorded before running, per the project's rules.

| # | Prediction | Outcome |
|---|---|---|
| 1 | zero-shot free-form 30–50% | **45.8%** ✓ |
| 2 | unparseable 10–25% | 4.7% ✗ |
| 3 | constrained beats free-form by 10–20 pts | +1.3 pts ✗ |
| 4 | few-shot k=77 beats zero-shot by 10–20 pts | lost by 40 ✗ |
| 5 | exemplar-set collapse caps accuracy at k/77 | refuted ✗ |
| 6 | recency anchoring in 3 of 3 seeds | 1 of 3 ✗ |
| 7 | query sensitivity: zero-shot 25–40 distinct, k=20 under 10 | 30 / 10,5,6 ✓ |

**Two of seven.** The largest miss (#4) was wrong in sign as well as
magnitude. Prediction #3 in particular means the format-effect framing in
`docs/02_evaluator.md` overstates its case by roughly tenfold; the
three-regime reporting still earns its place, because measuring the effect
is how we know it is small.

The best method in the project — kNN — was not in the plan, has no trained
parameters, and runs on a laptop.

## Artefacts

```
src/03_baselines.py        prompting baselines, resumable, both regimes
src/03c_knn_baseline.py    kNN, k chosen on dev
src/03d_prior_probe.py     query-sensitivity and prior probe
src/compare.py             paired McNemar between any two runs
src/inspect_prompts.py     prompt rendering and cross-seed collapse table
src/analyse_fewshot.py     exemplar-set membership analysis
reports/runs/<config>__<regime>/   predictions, env manifest, meta
```
