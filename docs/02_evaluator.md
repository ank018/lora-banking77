# Stage 2 — The evaluator

Written before any model existed to be evaluated, and before any baseline
was run. The ordering is the point: an evaluator tuned after seeing model
output is an evaluator tuned to flatter that output.

## Why this is not `pred == gold`

For a classifier it would be. But a decoder-only language model does not
emit a label — it emits text, and something has to decide what that text
meant. Asked to classify *"I am still waiting on my card?"*, a 1.5B model
can return any of:

```
card_arrival                          the contract, honoured
Card Arrival                          right answer, wrong shape
The intent is card_arrival.           right answer, wrapped in prose
card arrival / card_delivery_estimate a hedge, not an answer
waiting_for_card                      a label that does not exist
I'm not sure what they mean.          no answer at all
```

Every one needs a verdict, and the leniency of that decision *is* a
measurement decision. An evaluator can fail two ways, and neither is
visible in its own output:

- **too lenient** — wrong answers score correct, inflating everything
- **too strict** — right answers are discarded, deflating everything

So the test suite runs in both directions. Twelve tests prove it accepts
answers that are right but cosmetically odd; ten prove it refuses answers
that look right and are not. 31 tests total.

## Three-way verdicts

| Verdict | Meaning |
|---|---|
| `correct` | recovered label equals gold |
| `wrong_label` | recovered a real, different label |
| `unparseable` | no single valid label could be recovered |

`unparseable` is **not** a kind of wrong answer. Merging it into
`wrong_label` would erase the output-format story entirely, and that story
is one of the three confounds this project exists to measure.

## Match modes: evidence, not a score

Each row records *how* the label was recovered:

| Mode | Meaning |
|---|---|
| `exact` | the canonical label string, nothing else |
| `normalized` | matched after case/separator normalisation |
| `extracted` | a label found inside surrounding prose |
| `ambiguous` | two or more distinct labels present; refused |
| `none` | no valid label present |
| `constrained` | argmax over the label set; parsing not involved |

Storing the mode rather than a verdict alone means the sensitivity of any
number to the parser's generosity is recomputable later without rerunning a
single generation. Same principle as stage 1 storing near-duplicate
similarities instead of a boolean flag.

## Two decoding regimes

**Free-form** — the model generates; we parse. Measures classification
skill *and* format compliance, mixed together.

**Constrained** — the model scores all 77 labels and we take the argmax.
Invalid output is impossible, so `unparseable` cannot occur. Measures
classification only.

Every configuration is evaluated under both. The gap between them is a
reported number: how much of a fine-tuning gain was the model learning to
format rather than learning to classify. Base models pad their answers and
fine-tuned models do not, so this gap runs systematically in favour of
fine-tuning — which is exactly why it gets measured rather than assumed
away.

## Two parser hazards found by testing, not by reading

**Nested labels.** Three Banking77 labels are substrings of others:
`card_not_working` inside `virtual_card_not_working`, and `exchange_rate`
inside `card_payment_wrong_exchange_rate` and
`wrong_exchange_rate_for_cash_withdrawal`. Naive substring extraction sees
two labels in `virtual_card_not_working` and refuses it as ambiguous.
Longest match wins; both directions are tested.

**Query echo.** 5.5% of Banking77 test queries contain a label as a
substring — *"I need to know your exchange rates"* contains
`exchange_rate`. A model that repeats the question before answering would
be credited for its own echo. Since base models echo constantly and
fine-tuned models stop almost immediately, this bias runs in the familiar
direction. The parser now strips the query before extraction.

This one was invisible in code review. It surfaced only from running the
parser against all 3,080 real test queries and counting how many parsed to
a label unaided.

Residual limitation: a *paraphrased* echo still slips through. Unquantified,
and one more reason every number is rescorable under strict matching.

## Reporting conventions

These are fixed now, before any baseline runs, because both change what
every table in the project means.

**Lenient is the headline; strict is reported beside it.** `extracted`
matches count as correct by default. The reasoning is not that leniency is
more accurate but that it is *conservative with respect to our own
hypothesis*: forgiving prose padding helps base models most and shrinks the
fine-tuning gap. Choosing the convention that makes the expected result look
weaker is the right default. Strict scores are derivable from stored records
at no cost, and both will be published.

**`unparseable` stays in the denominator.** Accuracy is `correct / all`. A
system that emits garbage did not answer the question, and the person
waiting on a routed ticket does not care why. Parse rate is reported as its
own metric.

The obvious objection — that this understates classification ability — is
already answered by the design: **constrained decoding is the number that
isolates classification skill.** The two regimes divide the work between
them, so free-form has no need to invent a conditional-accuracy convention
to compensate. Conditional accuracy over parseable outputs remains
computable from stored records for anyone who wants it.

Together the three points form a ladder of increasing format strictness:

```
constrained            no format penalty at all
free-form, lenient     prose tolerated, label must be present and unique
free-form, strict      the output contract, honoured exactly
```

Differences between adjacent rungs are format effects. Differences that
survive all three are not.

## Artefacts

```
src/evaluator.py          parsing, verdicts, aggregation, strict rescoring
tests/test_evaluator.py   31 tests, both directions
```

Every generation is stored with `id`, `query`, `raw`, `pred`, `gold`,
`match_mode` and `verdict`, so every convention above can be revisited
after the fact without spending a single GPU-second.
