# Stage 5 — The encoder baseline

The comparison most write-ups skip: a small, conventional classifier
trained the old way, on exactly the data the fine-tuned LLM will get.

It was planned as a supporting baseline. It took two days, produced a model
that could not be trained at all, and exposed a comparison this project had
been making unfairly. Its purpose is to give the LoRA numbers a scale, and
to answer one question the LoRA experiment cannot answer alone: **how much
of fine-tuning's advantage comes from the pretrained model, and how much
from having labelled data at all.**

## Results

RoBERTa-base (125M), fp32, lr 5e-5, batch 16, 30 epochs, best epoch
selected on dev. Three seeds per rung; test never touched during training
or model selection.

| Examples | per class | kNN | **RoBERTa** | gap | seed sd |
|---:|---:|---:|---:|---:|---:|
| 154 | 2 | 48.2% | **56.5%** | +8.3 | 1.62 pp |
| 308 | 4 | 51.4% | **71.0%** | +19.6 | 0.47 pp |
| 616 | 8 | 61.6% | **82.3%** | +20.7 | 0.39 pp |
| 1,232 | 16 | 67.9% | **87.6%** | +19.7 | 0.33 pp |
| 1,848 | 24 | 72.8% | **89.5%** | +16.7 | 0.38 pp |
| 9,387 | full | 82.9% | **94.0%** | +11.1 | 0.23 pp |

Both columns are scored on the same frozen 3,080-item test set. kNN is
restricted to the same training examples at each row.

**Training beats lookup at every data size tested.** There is no crossover
point. The gap is largest in the middle of the curve and narrows at both
ends — at 2 examples per class neither method has much to work with, and by
9,387 both are approaching what the label noise allows.

The full-pool row is a three-seed mean: 94.25 / 93.90 / 93.83.

**RoBERTa matches full-pool kNN using 6.6% of the data.** 82.3% on 616
examples against kNN's 82.9% on 9,387: delta −0.71 pp, McNemar p = 0.39,
CI [−2.26, +0.83]. On the clean subset it is +0.04 pp at p = 1.00.
Indistinguishable.

At equal data the same comparison is not close: rung 8, +20.55 pp,
p ≈ 10⁻¹²⁰, 731 items fixed against 98 broken.

### The unequal comparison, and its correction

For several stages this project reported kNN at 82.9% as "the bar to beat"
and treated RoBERTa's 82.3% at rung 8 as merely matching it. That
comparison was wrong: **kNN was being given the full 9,387-example pool
while the encoder was trained on 616.** A fifteen-fold data advantage,
introduced by the way the baselines were built and repeated without being
checked.

`src/04b_knn_baseline.py --rungs` now restricts kNN's reference set to each
training rung, so both methods share an x-axis. The full-pool kNN figure is
still reported, because "what you get from all your labelled data with no
model" is a real question — but it is no longer set against models trained
on a fraction of it.

One parameter changed in that rewrite: TF-IDF `min_df` from 2 to 1, since
at 154 examples a minimum document frequency of 2 discards most features.
Full-pool figures moved by at most 0.1 point (`knn_k1` 80.2 → 80.1, `knn_k5`
unchanged), and one setting now applies at every rung.

## Training-seed noise, measured

Stage 0's sizing simulation predicted that **training-seed variance, not
test-set size, would be the binding constraint** on every ablation in this
project, and assumed 1–3 pp.

Measured: **0.33–0.47 pp** at rung 4 and above. Roughly a third of the
pessimistic estimate.

| per class | seed sd | range across 3 seeds |
|---:|---:|---:|
| 2 | 1.62 pp | 3.12 pp |
| 4 | 0.47 pp | 0.94 pp |
| 8 | 0.39 pp | 0.71 pp |
| 16 | 0.33 pp | 0.65 pp |
| 24 | 0.38 pp | 0.75 pp |

Variance shrinks as data grows, which is the sensible direction, and is
largest where the model has least to learn from.

At the full pool, three seeds: **0.23 pp** (94.25 / 93.90 / 93.83).
LoRA on Qwen3-1.7B measured **0.26 pp** at the same rung — near-identical,
where I had expected generation-based training on a 2B model to be noisier.

**Consequence.** With three seeds the standard error of a mean is about
0.15 pp, so differences around half a point become detectable at full data.
The constraint the project was designed around turned out much looser than
feared.

One caveat carried into stage 7: this is seed noise **at the full pool**.
At rung 8 the ablation runs measured a pooled sd of **0.79 pp**, three
times larger, and the ablation script's footer initially quoted the
full-pool figure at rung 8 — making its significance threshold roughly
three times too generous. Noise is a property of a regime, not of a
project.

## DeBERTa-v3 could not be trained on this stack

The encoder was originally `microsoft/deberta-v3-base`. It never learned
anything. The diagnosis took four escalating steps, and each ruled
something out:

**1. fp16 overflow.** `AutoModelForSequenceClassification.from_pretrained`
loaded the model in float16 — we never specified a dtype and transformers
took it from the checkpoint. DeBERTa-v3's disentangled attention overflows
in half precision: loss starts correctly at ln(77) = 4.34 and is NaN within
ten steps, at every learning rate tested. Fixed by passing
`dtype=torch.float32` explicitly, plus an assertion.

That fixed the NaNs and not the training.

**2. The loop can learn.** All three of deberta-v3, roberta-base and
bert-base-uncased memorise 32 examples perfectly — loss 4.32 → 0.0003, 100%
train accuracy in 300 steps. Gradients flow; the optimiser steps.

A caveat worth stating: memorising 32 examples proves the plumbing works.
It does **not** prove the labels are correct — a model memorises arbitrary
text→label pairs just as happily. That test was necessary and not
sufficient, and it was written as though it were both.

**3. The scheduler is not to blame.** Four configurations on real rung-8
data with the learning rate logged as the optimiser sees it: with and
without linear warmup, at 2e-5 and 5e-5. All four failed, and the learning
rate was exactly what it should be. Hypothesis refuted.

**4. Model, not pipeline.** 1,000 steps on identical data through identical
code:

```
roberta-base    step 100: dev 46.9%   step 500: dev 84.3%   step 1000: dev 84.4%
deberta-v3      step 100: dev  1.3%   step 500: dev  2.4%   step 1000: dev  2.3%
```

Chance is 1.3%.

**Signature:** deberta-v3 memorises 32 examples at 100% yet generalises at
chance. A model that can fit arbitrary targets but cannot transfer anything
from pre-training has broken representations, not a broken training setup.
Most likely a deberta-v3 / transformers 5.0 incompatibility. Not chased
further — this is a supporting baseline and it had already cost two days.

roberta-base is 125M against Qwen3's 2.03B, so the "much smaller model"
comparison is unaffected.

### What made this expensive

**None of these failures raised an error.** fp16 produced a plausible
number (1.3%). The chance-level runs completed successfully and wrote
fifteen result directories. The first version of the training script logged
no per-epoch loss, so fifteen runs and thirty minutes of GPU produced no
evidence about why anything failed — the missing log line was the real bug,
and the training failure was downstream of it.

The script now prints loss and dev accuracy per epoch, raises on non-finite
loss, warns loudly when a run finishes at chance, and stores the full loss
history in each run's `meta.json`.

Three framework defaults have now been wrong for this project and silent
about it: `datasets` dropping script loading, `torch.cuda.is_bf16_supported()`
returning True on a T4, and `from_pretrained` choosing fp16.

## The full-pool row is not a rung

9,387 examples at the natural class distribution — 27 to 179 per class,
5.3× imbalance — where every other row is class-balanced. It varies
**size and balance at once**, so it is reported as its own row rather than
as an extension of the controlled curve. It answers "what if you use all
your labelled data", which is a real question, not "what does the next
point on the scaling curve look like".

It also ran **15 epochs rather than 30**. At 9,387 examples that is 8,805
gradient steps, already more than double what rung 24 received, and best
epoch landed at 11 of 15. Thirty would have cost 35 extra minutes for
nothing. The inconsistency is deliberate and recorded here rather than
hidden.

Dev accuracy after one epoch was 80.4%, against 1.3% at rung 8 — with 587
steps per epoch instead of 39, the cold-start phase is over before the
first evaluation. That cold start is also why "flat loss after 40 steps"
was not diagnostic for DeBERTa: a working run looks identical for the first
two epochs on this task.

## The bar this sets

**94.0% ± 0.23 from a 125M model, 30 minutes on a free T4, 64 prompt tokens
per prediction, no label list, no generation.**

Qwen3-1.7B is 2.03B parameters — 16× larger — and its constrained
evaluation costs roughly 330 ms per item against RoBERTa's few
milliseconds.

**Outcome (stage 6):** LoRA reached 93.59% ± 0.26 at matched epoch budgets.
The 0.40-point difference is not significant (seed-level p = 0.119;
item-level p = 0.31–0.38 across three matched seeds). But at **154
examples** LoRA leads by **+8.93 pp** (p ≈ 10⁻¹⁸). The encoder's role in
this project is to locate that crossover — somewhere between 308 and 616
examples — not to win a contest.

## Predictions, scored

| Prediction | Outcome |
|---|---|
| DeBERTa 88–93% at largest rung | untrainable ✗ |
| seed sd 0.5–2.0 pp | 0.33–1.62 ✗ (mostly below) |
| curve still climbing at rung 24 | yes ✓ |
| encoder crosses kNN between rungs 8 and 16 | no crossover; wins everywhere ✗ |
| RoBERTa 86–89% at rung 24 | 89.5% ✗ (just above) |
| RoBERTa 91–93% at full pool | 94.0% ✗ (above) |

**One of six.** Every miss was in the same direction: the small
conventional model is better, and more data-efficient, than predicted.

The largest, though, is the first row. Predicting a model's accuracy is
guesswork; predicting that the model would be **untrainable on the target
stack** was not something any amount of care would have produced. That
failure mode — a framework default silently destroying a model — is not on
the list of things a plan anticipates.

## Limitations

- The full-pool row is a three-seed mean (94.25 / 93.90 / 93.83, sd 0.23).
  Every other row is also three seeds; the kNN column is deterministic.
- Hyperparameters (lr 5e-5, 30 epochs, batch 16, max_len 64) were fixed
  before the first run and never tuned per rung. Tuning per rung would
  confound "more data helps" with "this rung got better settings".
- Fixed epochs means the smallest rung sees 300 gradient steps and rung 24
  sees 3,480, so the curve partly reflects optimisation budget as well as
  data quantity. Best-epoch-on-dev mitigates this; it does not remove it.
- Only one encoder is reported. bert-base-uncased was verified to train on
  this stack but not swept.

## Artefacts

```
src/05a_encoder_baseline.py    the sweep; --rungs 0 means the full pool
src/04b_knn_baseline.py        kNN, k on dev, --rungs for the matched curve
src/05b_diagnose_encoder.py    tokenizer, dtype, first forward pass, 40 steps
src/05c_overfit_test.py        can three encoders memorise 32 examples
src/05d_isolate_training.py    scheduler and learning rate, four configs
src/05e_model_vs_pipeline.py   roberta vs deberta, 1,000 steps, real data
reports/runs/robertabase_rung*_seed*__constrained/
reports/runs/knn_rung*__constrained/
```

The four diagnostics are kept rather than deleted. They are the evidence
for the DeBERTa claim, and any future change of model, dtype or
transformers version needs them again.
