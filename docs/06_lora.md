# Stage 6 — LoRA fine-tuning

The experiment this project was built for. Qwen3-1.7B, LoRA rank 16 on all
attention projections and the MLP triple, trained on nested subsets of a
frozen 9,387-example pool and scored on 3,080 held-out items under both
decoding regimes.

Everything before this stage exists to make these numbers interpretable.

## Does the adapter work?

Yes, and the effect is enormous. Measured with the **prompt held constant**
at 63 tokens, so the adapter is the only thing that differs:

| | free-form | constrained |
|---|---:|---:|
| base Qwen3-1.7B, bare prompt | 1.7% | 31.4% |
| **LoRA, bare prompt, full pool** | **93.7%** | **93.9%** |
| **adapter effect** | **+92.0 pp** | **+62.5 pp** |

Paired, constrained: **+62.53 pp, 95% CI [+60.76, +64.31]**, McNemar
p below floating-point resolution. 1,954 items fixed against 28 broken —
a ratio of 70 to 1, across 1,982 discordant items.

**The two figures measure different things and both belong here.** The
constrained number is what fine-tuning taught the model about the *task* —
telling 77 banking intents apart — with output formatting removed from the
picture. The free-form number is task *and* output contract together. The
30-point gap between them is the size of the formatting problem in the
untuned model, and it is only visible because stage 2 fixed two decoding
regimes before any model was run.

### Why the prompt has to be held constant

The obvious before/after — zero-shot 47.1% against LoRA 93.6% — is
confounded. Zero-shot uses a 447-token prompt listing all 77 labels; the
fine-tuned model uses 63 tokens and no list. That comparison mixes the
adapter with the loss of the label list.

Running the untuned model on the bare prompt separates them:

- label list is worth **+15.7 pp** to the untuned model (31.4% → 47.1%)
- the adapter is worth **+62.5 pp** at a constant prompt

So the naive +46.5 pp *understates* the adapter, because the fine-tuned
model is also being denied 384 tokens of context the baseline received.

## How much data does it need?

Constrained accuracy on test, seed 1, 8 epochs, best epoch chosen on dev:

| examples | per class | LoRA | free-form | unparseable |
|---:|---:|---:|---:|---:|
| 154 | 2 | 66.7% | 50.0% | 34.9% |
| 308 | 4 | 74.8% | 66.8% | 16.5% |
| 616 | 8 | 81.2% | 79.6% | 5.0% |
| 1,232 | 16 | 86.7% | 85.1% | 3.0% |
| 1,848 | 24 | 89.3% | 88.5% | 1.7% |
| 9,387 | full | 93.9% | 93.7% | 0.5% |

Two examples per class already reaches 66.7% on a 77-way problem where
chance is 1.3% and the untuned model manages 31.4%.

## Output format is learned slowly, not quickly

`docs/02_evaluator.md` asserted that fine-tuning fixes formatting "almost
immediately, in the first few dozen examples, because that's the easiest
thing for it to learn."

That was wrong, and the curve shows it:

| examples | 0 | 154 | 308 | 616 | 1,232 | 1,848 | 9,387 |
|---|---:|---:|---:|---:|---:|---:|---:|
| unparseable | 96.1% | 34.9% | 16.5% | 5.0% | 3.0% | 1.7% | 0.5% |
| free-form vs constrained | 29.7 | 16.7 | 8.0 | 1.6 | 1.6 | 0.8 | 0.2 |

At 154 training examples — every one of which had a clean bare label as its
target — **a third of the model's outputs still cannot be parsed.** Format
compliance takes thousands of examples, not dozens, and it decays smoothly
rather than snapping into place.

This also revises the stage 4 finding. The format effect measured on the
*base* model was 1.3 points, which I described as small. It is small at
either extreme — an untuned model with a label list, or a model trained on
9,387 examples — and large in the middle, peaking at 16.7 points for a
model trained on 154. The two-regime design earns its cost in that middle
region, which is exactly where a practitioner with limited labels operates.

## Near-duplicate memorisation: predicted, and absent

Stage 1 found that 13.8% of test items have a same-label near-twin in the
training pool, and predicted:

> Full-test accuracy will exceed clean-subset accuracy for every fine-tuned
> config, and **the gap will widen with rung size** — a fine-tuned model can
> recall those twins and prompting cannot, so the effect grows with data.

Every accuracy in this project was reported twice on the strength of that.
The gap **narrows**:

| examples | full | clean | gap |
|---:|---:|---:|---:|
| 154 | 66.7% | 65.1% | 1.6 |
| 308 | 74.8% | 73.3% | 1.5 |
| 616 | 81.2% | 79.7% | 1.5 |
| 1,232 | 86.7% | 85.8% | 0.9 |
| 1,848 | 89.3% | 88.3% | 1.0 |
| 9,387 | 93.9% | 93.4% | **0.5** |

roberta-base shows the same shape (0.8 → 0.5), and so does **zero-shot**,
at 1.0 — a configuration that has no training set and therefore cannot
memorise anything at all.

So the gap is not memorisation. It is **intrinsic difficulty**:
near-duplicated queries are common phrasings with many variants, and they
are the easy ones. Every method finds them easier, including one with no
training data, and the advantage shrinks as models get good enough to
solve them anyway.

The convention still earned its cost — dual reporting is how we know the
effect is absent, and the alternative was asserting it after the fact — but
the concern that motivated it did not materialise, and no headline in this
project needs adjusting for recall.

## Training-seed noise

Full pool, 8 epochs, three seeds: **93.90 / 93.47 / 93.41 → 93.59%,
sd 0.26 pp.**

Nearly identical to roberta-base's 0.23 pp. I had expected generation-based
training on a 2B model to be noisier than a classification head on a 125M
one. It is not.

Stage 0's simulation assumed seed noise of 1–3 pp and warned it would bind
every comparison in the project. Measured at roughly a quarter of that,
which is why ~0.6 pp differences are resolvable with three seeds.

## The epoch budget, and a confound we removed

The first full-pool runs used 4 epochs, chosen to save quota, with dev
accuracy still climbing at the last epoch. roberta-base ran 15 epochs and
selected a dev peak at epoch 11 — it was not similarly constrained.

That asymmetry ran in the direction of our conclusion, so it was removed
rather than noted. Rerunning all three seeds at 8 epochs:

| | full test |
|---|---:|
| LoRA, 4 epochs, 3 seeds | 93.44% |
| LoRA, 8 epochs, 3 seeds | 93.59% |

The extra epochs were worth **+0.17 pp**, and nearly all of it came from
seed 1 (93.60 → 93.90); seeds 2 and 3 barely moved. So the confound was
mostly seed noise dressed up as a systematic handicap.

**It changed the reported conclusion anyway.** With mismatched budgets the
encoder comparison read p = 0.036 — significant. With matched budgets it
reads p = 0.119 — not. The published claim would have rested on an
asymmetry we knew about, and no reader could have found it. Three hours of
GPU to avoid that was the right trade.

## Context: is 93.9% any good?

Six reference points, same frozen test set, so the LoRA numbers have a
scale. These are context for the experiment above, not the experiment.

| Method | Accuracy | Data used |
|---|---:|---|
| majority class | 1.3% | — |
| few-shot, 77 exemplars | 5.7% | 77 |
| few-shot, 5 exemplars | 22.0% | 5 |
| base model, bare prompt | 31.4% | 0 |
| zero-shot, 77 labels listed | 47.1% | 0 |
| kNN over TF-IDF | 82.9% | 9,387 |
| roberta-base, fully fine-tuned | 94.0% ± 0.23 | 9,387 |
| **LoRA on Qwen3-1.7B** | **93.6% ± 0.26** | **9,387** |

### When is LoRA worth it?

Against roberta-base at matched data, one seed each (three at the full
pool):

| examples | LoRA | roberta | gap |
|---:|---:|---:|---:|
| 154 | **66.7%** | 56.5% | **+10.2** |
| 308 | **74.8%** | 71.0% | **+3.8** |
| 616 | 81.2% | 82.3% | −1.1 |
| 1,232 | 86.7% | 87.6% | −0.9 |
| 1,848 | 89.3% | 89.5% | −0.2 |
| 9,387 | 93.6% | 94.0% | −0.4 |

At 154 examples the gap is **+8.93 pp** paired (CI [+6.95, +10.90],
p ≈ 10⁻¹⁸, 631 items fixed against 356 broken) — six times roberta's seed
noise at that rung. At the full pool the gap is **−0.40 pp**, not
significant on any of five tests (seed-level p = 0.119; item-level
p = 0.31–0.38 across three matched seeds).

**The crossover is somewhere between 308 and 616 examples.**

The reading: a pretrained model's knowledge substitutes for labelled data,
and that substitution is worth about nine points at two examples per class
and nothing at all beyond eight. Discordance tells the same story — the two
models disagree on 32% of test items at 154 examples and 4.4% at 9,387.
With little data they behave differently; with enough they converge on the
same answers.

## Configuration

Fixed before the first run. Stage 7 varies one factor at a time from here,
so those are honest variations rather than a search reported afterwards as
an ablation.

```
model         Qwen/Qwen3-1.7B (2.03B parameters), fp16
rank          16          alpha 32          dropout 0.05
targets       q,k,v,o_proj + gate,up,down_proj   (17.4M trainable, 0.85%)
lr            2e-4, linear schedule, 10% warmup
batch         8 x 2 grad accum = 16 effective
max_len       96          epochs 8
loss          label tokens only; prompt masked to -100
selection     best epoch on dev, free-form; test touched once per config
prompt        bare, 63 tokens, no label list
```

**LoRA parameters are kept in fp32** while the base stays fp16. Adam
moments on fp16 parameters underflow, and the T4 has no bf16 to fall back
on (`docs/00_environment.md`).

Stage 7 varies rank, learning rate and target modules one at a time from
this configuration (`docs/07_ablations.md`). In short: **the MLP modules
matter far more than rank** — dropping `gate/up/down_proj` costs 7.8 points
— and rank 64 is worth +3.35 points at 616 examples but **−0.42, not
significant, at 9,387**. Capacity substitutes for data, exactly as the
pretrained model's knowledge does.

**The bare prompt is a deliberate handicap.** The fine-tuned model receives
strictly less prompt context than every baseline it is compared against —
63 tokens against zero-shot's 447. It also makes the cost comparison
honest, since a deployed fine-tuned model would not pay for a label list it
no longer needs. Same principle as choosing lenient parsing for the
headline in stage 2: where two conventions are defensible, take the one
that makes the expected result look weaker.

## Cost

| | training | inference | prompt | parameters |
|---|---:|---:|---:|---:|
| roberta-base | 30 min | ~3 ms/item | 64 tok | 125M |
| LoRA Qwen3-1.7B | 79 min | 330 ms/item | 63 tok | 2.03B (17.4M trained) |

Roughly 2.6× the training time and 100× the inference time for the same
accuracy at full data. Matching the epoch budget made LoRA's cost
disadvantage larger, not smaller — and rank 64, which buys nothing at full
data, costs 94 minutes rather than 79.

## Predictions, scored

| Prediction | Outcome |
|---|---|
| LoRA 91–95% at full pool | 93.6% ✓ |
| overlaps roberta rather than clearly beating it | p = 0.12 ✓ |
| fine-tuning fixes formatting within dozens of examples | takes thousands ✗ |
| LoRA seed noise larger than roberta's | 0.26 vs 0.23 ✗ |
| base model on bare prompt under 10%, unparseable above 40% | 1.7%, 96.1% ✗ (right direction, wrong magnitude) |
| *(stage 1)* full-vs-clean gap widens with rung size | narrows, 1.6 → 0.5 ✗ |
| *(stage 7)* rank differences inside noise | +3.35 pp at 616 examples ✗ |
| *(stage 7)* attention-only within ~1 pp of attention+MLP | −7.79 pp ✗ |

## Limitations

- Rung results are **single-seed**; only the full pool has three. The
  crossover point is therefore located between 308 and 616 examples rather
  than pinned precisely.
- **One base model, one task, one label space.** Nothing here establishes
  where the crossover sits for a different model size or a task with
  fewer classes.
- The comparison uses roberta-base as the encoder because deberta-v3 could
  not be trained on this stack (`docs/05_encoder.md`). Other encoders were
  not swept.
- Epoch budgets are matched in spirit, not exactly: LoRA runs 8 epochs and
  roberta 15, each selecting its best epoch on dev. Both had plateaued.
- Inference timings are single-item constrained scoring on a T4 and are
  indicative, not a serving benchmark.
- **Dev predictions were never persisted.** The training loop scores dev
  every epoch but writes only test predictions to disk, so the stage 8
  failure taxonomy has to read test failures rather than dev ones. Saving
  both would have cost nothing and should have been in the loop from the
  first run.

## Artefacts

```
src/04_train_lora.py      training, both eval regimes, --tag for variants
src/prompts.py            bare() and SYSTEM_BARE, versioned
src/compare_families.py   seed-level and item-level tests between procedures
reports/runs/qwen3lora_*  predictions, env manifest, full loss history
reports/runs/bare__*      the untuned control at a constant prompt
```
