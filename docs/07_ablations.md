# Stage 7 — Ablations

One factor varied at a time from a centre configuration fixed before any
LoRA run happened: r=16, alpha=32, dropout 0.05, lr 2e-4, targeting the
four attention projections plus the MLP triple. Because the centre was
chosen in advance, these are variations rather than a search reported
afterwards as an ablation.

Scored on **dev** at rung 8 (616 examples), free-form, three seeds each.
Rank and learning rate are choices, and choices are made on dev. It is also
5× cheaper: 616 items by generation takes seconds against 19 minutes for a
constrained pass over 3,080.

## Results

| config | dev mean | sd | vs centre | trainable | min/run |
|---|---:|---:|---:|---:|---:|
| centre (r=16) | 81.06% | 1.51 pp | — | 17.4M | 8.5 |
| rank 8 | 81.87% | 0.61 pp | +0.81 | 8.7M | 8.3 |
| **rank 64** | **84.42%** | 0.43 pp | **+3.35** | 69.7M | 9.0 |
| lr 1e-4 | 79.71% | 0.59 pp | −1.35 | 17.4M | 8.4 |
| lr 5e-4 | 82.52% | 0.66 pp | +1.46 | 17.4M | 8.3 |
| **attention only** | **73.27%** | 0.41 pp | **−7.79** | 6.4M | 6.7 |

### The significance threshold, and a mistake in it

The script's footer originally quoted seed noise of **0.23–0.26 pp**, taken
from stage 5's full-pool measurement, and concluded that ~0.6 pp
differences were resolvable. That is the wrong number for this regime.

**Noise is a property of a regime, not of a project.** These runs are at
616 examples, where the encoder sweep already measured seed sd around
0.4 pp and where the centre configuration here measures **1.51 pp**. Pooled
across the six configurations, sd ≈ **0.79 pp**, so the standard error on a
difference of two three-seed means is ~0.65 pp and significance at df ≈ 4
needs roughly **1.8 pp** — three times the threshold printed.

Under the correct threshold: **rank 64 and attention-only are significant;
rank 8, lr 1e-4 and lr 5e-4 are not.**

The centre's own sd of 1.51 pp is a weakness in the design. Every delta is
measured against a noisy reference, driven by one seed at 79.38% against
82.31% and 81.49%. More centre seeds would tighten every row in the table.

## The MLP modules do most of the work

Dropping `gate_proj`, `up_proj`, `down_proj` and adapting attention alone
costs **7.79 points** — larger than any other factor tested, and the only
change that is unambiguously significant against even a generous threshold.

Most LoRA guidance targets attention projections only. On this task that
choice is worth more than a 4× change in rank in either direction, and it
is the one parameter that is not usually presented as a choice at all.

It is also the cheapest configuration — 6.4M trainable parameters and
6.7 minutes per run against 8.5 — so the saving is real and so is the cost
of taking it.

## Rank capacity substitutes for data

Rank 64 is worth +3.35 points at 616 examples. At the full pool it is
worth nothing:

| examples | r=16 | r=64 | delta | |
|---:|---:|---:|---:|---|
| 616 (dev, free-form) | 81.06% | 84.42% | **+3.35** | significant |
| 9,387 (test, constrained) | 93.90% | 93.47% | −0.42 | p = 0.27, n.s. |

Paired at full data: 117 discordant items of 3,080, CI [−1.11, +0.27].
Four times the adapter capacity, 94 minutes of training against 79, and no
measurable difference.

**The two rows are not directly comparable** — different split, different
decoding regime — but the comparison *within* each row is valid, and the
sign flips.

This is the same shape as every other effect in this project: **large when
data is scarce, absent when it is not.** The pretrained model's knowledge
substitutes for labelled data (`docs/06_lora.md`, +8.93 pp at 154 examples
and −0.40 at 9,387); adapter capacity does the same. Whatever supplies the
missing information — prior knowledge, parameters, retrieved neighbours —
stops mattering once there are enough labels.

It also means the headline configuration was not understated. r=16 was
chosen before any run and is the right choice at full data.

## Learning rate does not matter much

1e-4 → 2e-4 → 5e-4 spans −1.35 to +1.46 points around the centre, and
neither end is significant at the 1.8 pp threshold. The trend favours
higher, and 5e-4 showed no instability — no divergence, no NaN, best epochs
landing at 5–7 rather than earlier.

That is a mild refutation of the usual advice to tune learning rate first.
On this task the target-module choice was worth five times as much as the
learning-rate range tested.

## Predictions, scored

Written before the grid ran.

| Prediction | Outcome |
|---|---|
| rank differences inside noise; r=8 no worse than r=64 | r=64 +3.35 pp, significant ✗ |
| lr matters; 5e-4 unstable or worse, 1e-4 undertrained | neither significant; 5e-4 fine ✗ |
| attention-only within ~1 pp of attention+MLP | −7.79 pp ✗ |

**Zero of three**, and the third by a factor of eight. The one I was most
confident about — that rank would not matter — is wrong at small data and
right at large data, which is a distinction I did not think to make.

## Limitations

- **One rung.** Everything here is measured at 616 examples, except the
  r=64 full-pool check. Whether the MLP finding also decays with data is
  untested, and the rank result says it might.
- **Dev, free-form, one regime.** Constrained scoring would remove the
  formatting component, which at 616 examples is worth ~1.6 points
  (`docs/06_lora.md`) and could differ between configurations.
- **The centre is noisy** (sd 1.51 pp), so every delta inherits it.
- **Alpha scales with rank** — r=8/α=16, r=16/α=32, r=64/α=128 — holding
  α/r fixed. Without that, changing rank silently changes the adapter's
  effective learning rate and the ablation measures two things at once.
  This is a design choice, not a neutral one: an ablation holding α fixed
  instead would report different numbers.
- **Dropout, batch size, warmup ratio and max sequence length were not
  varied.** Nor was the epoch budget, which stage 6 showed is worth
  +0.17 pp at the full pool.

## Artefacts

```
src/05_ablations.py       one factor at a time; imports the real training loop
reports/ablations.json    per-seed dev accuracy and full loss history
reports/runs/qwen3lora_rungfull_r64ep8_seed1__*   the full-pool rank check
```

`05_ablations.py` imports `04_train_lora.py` rather than reimplementing
training, so the ablations exercise exactly the code that produced the
headline numbers.
