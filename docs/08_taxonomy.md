# Stage 8 — Failure taxonomy

Stage 1 wrote a prediction into `docs/01_dataset_design.md` before any
model existed:

> The largest failure cluster will be the intent pairs already visible in
> the probe — `verify_my_identity`/`why_verify_identity`,
> `card_arrival`/`card_delivery_estimate`, `exchange_charge`/`exchange_rate`
> — and they will persist at every rung, because more examples of a
> genuinely ambiguous boundary do not resolve it.

Every disagreement sample printed during stages 4–7 was one of those pairs.
This counts them instead of eyeballing them.

## The prediction, scored

**Persistence: confirmed.** The two main pairs hold a steady share of
errors across four configurations spanning a 60-fold range of training
data and two model architectures:

| pair | LoRA 154 | LoRA 616 | LoRA 9,387 | RoBERTa 9,387 |
|---|---:|---:|---:|---:|
| `verify_my_identity` / `why_verify_identity` | 3.3% | 2.8% | 2.7% | 2.3% |
| `card_arrival` / `card_delivery_estimate` | 2.0% | 1.9% | 4.8% | 2.8% |

Both stay in the top seven everywhere. Overall accuracy moves from 66.7% to
93.9% across that range; these pairs do not go away.

**"Largest cluster": overstated.** The biggest single pair accounts for
3–5% of errors, not a dominant share. Errors are spread over 111–335
distinct pairs depending on configuration.

**One of the three named pairs was wrong.** `exchange_charge` /
`exchange_rate` ranks 19th to 65th — 0.5–1.6% of errors. It was picked from
the stage-1 probe because the two labels are lexically similar. Lexical
similarity turned out not to predict which boundaries a model finds hard.

Verdict: **the mechanism was right, the magnitude and the specific
selection were not.** Which is the pattern across every mechanism
prediction in this project.

## Errors concentrate as data grows

| config | errors | distinct pairs | top-10 coverage |
|---|---:|---:|---:|
| LoRA 154 | 1,026 (33.3%) | 335 | 17.3% |
| LoRA 616 | 580 (18.8%) | 217 | 23.6% |
| LoRA 9,387 | 188 (6.1%) | 111 | 25.5% |
| RoBERTa 9,387 | 177 (5.7%) | 116 | 24.3% |

Of 2,926 possible label pairs, a model trained on 154 examples finds 335
ways to be wrong; one trained on 9,387 finds 111, and a quarter of its
errors fall on ten of them. **Data removes the long tail and leaves a hard
core.**

## What data fixes, and what it does not

The taxonomy records both directions of each confusion, and the asymmetry
is the interesting part.

**At 154 examples, the largest errors are one-way:**

```
unable_to_verify_identity  -> verify_my_identity                 23 : 0
card_payment_wrong_exchange_rate -> exchange_rate                14 : 0
beneficiary_not_allowed    -> failed_transfer                    13 : 0
balance_not_updated_after_bank_transfer -> ..._cheque_or_cash    15 : 0
```

A pure directional bias: the model has learned one label of the pair and
reaches for it. That is a model preference, and more examples cure it.

**At 9,387 examples, almost every surviving pair is symmetric:**

```
card_arrival <-> card_delivery_estimate                           9 : 3
top_up_failed <-> top_up_reverted                                 6 : 4
verify_my_identity <-> why_verify_identity                        5 : 2
balance_not_updated_after_bank_transfer <-> pending_transfer      5 : 3
```

Errors flow both ways. That is not a preference; it is a boundary the model
cannot locate because the boundary is not clearly there.

**Asymmetry is what training data fixes. Symmetry is what it leaves
behind.** The taxonomy makes that visible only because both directions are
counted, which a plain confusion count would not show.

## Two architectures fail on the same items

The full-pool top pairs for LoRA on Qwen3-1.7B and for roberta-base are
nearly the same set:

| pair | LoRA | RoBERTa |
|---|---:|---:|
| `card_arrival` / `card_delivery_estimate` | 4.8% | 2.8% |
| `top_up_failed` / `top_up_reverted` | 3.2% | 2.8% |
| `balance_not_updated_after_bank_transfer` / `pending_transfer` | 2.7% | 3.4% |
| `verify_my_identity` / `why_verify_identity` | 2.7% | 2.3% |
| `balance_not_updated_after_bank_transfer` / `transfer_not_received` | 2.1% | 2.8% |
| `exchange_via_app` / `fiat_currency_support` | 2.1% | 1.7% |

A 2.03B decoder fine-tuned by LoRA and a 125M encoder fine-tuned
conventionally — different architectures, different objectives, different
prompts — converge on the same residual errors.

**That is the strongest evidence in this project that the remaining errors
belong to the data rather than to the models.** It also explains the
stage-6 result: two methods that fail on the same items cannot be far apart
in accuracy, which is why they disagree on only 4.4% of the test set at
full data and why their 0.40-point difference is not significant.

### A cluster, not a pair

`balance_not_updated_after_bank_transfer` appears in four of the top pairs
at full data — with `pending_transfer`, `transfer_timing`,
`transfer_not_received_by_recipient`, and
`balance_not_updated_after_cheque_or_cash_deposit`. It is not one confusable
neighbour but a **transfer-status cluster** where several intents describe
overlapping situations: money sent, money not arrived, balance not
reflecting it, how long it should take. A customer message often satisfies
several of those at once.

## Does the labelled data itself disagree?

Using stage 1's near-duplicate evidence: dev items whose nearest training
twin at cosine ≥ 0.85 carries a **different** label.

```
4 items total
  3   verify_my_identity        <-> why_verify_identity
  1   card_payment_not_recognised <-> direct_debit_payment_not_recognised
```

Only four, and three of them are the pair stage 1 named first.

That number is small for the reason stage 1 flagged: **TF-IDF character
n-grams see lexical similarity, not semantic similarity.** Two differently
worded queries about the same thing are invisible to it. Four is a floor on
annotation inconsistency, not an estimate of it — and the symmetric error
pattern above suggests the real figure is much larger.

## A design error this stage exposed

**The taxonomy was supposed to run on dev. It could not.**

Stage 1 reserved dev for exactly this — "error taxonomy, prompt iteration,
anything that looks at failures" — so that test could be scored once per
frozen configuration and never inspected. But the training scripts evaluate
dev every epoch and write only *test* predictions to disk. No per-item dev
predictions exist for any trained model.

Reconstructing them means retraining eighteen models, since
`--save-adapter` was never used and no checkpoints were kept.

So this analysis reads **test** predictions, and the script labels every
table accordingly. The mitigating facts: it is descriptive, no modelling
decision follows from it, and every configuration analysed was already
scored and frozen. It is still test inspection, and the rule stage 1 set
was not honoured.

The fix costs nothing and should have been in the training loop from the
first run: **persist dev predictions alongside test.** Dev is already being
generated every epoch; only the per-item results are thrown away.

## Limitations

- Read on test, not dev, for the reason above.
- Single seed per configuration. Pair-level counts of 3–9 errors are small,
  and the ordering within the top ten is not stable — only the persistence
  of the leading pairs is.
- `unparseable` predictions are grouped under a single `<unparseable>`
  pseudo-label, so at rung 2 — where a third of free-form outputs are
  unparseable — the constrained predictions are used and that failure mode
  is invisible here. It is characterised in `docs/06_lora.md` instead.
- Pairs are unordered for counting and directional for the symmetry check;
  a three-way confusion appears as three pairs.

## Artefacts

```
src/07_taxonomy.py    confusion pairs, symmetry, persistence, data disagreement
```
