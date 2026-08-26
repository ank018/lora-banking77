# When is LoRA worth it?

Fine-tune a 2-billion-parameter language model to sort banking messages
into 77 intents, and it scores 93.6%. Train a 125-million-parameter encoder
on the same data and it scores 94.0%.

The interesting question is not which number is bigger — they are
statistically indistinguishable — but **at what point the larger model
stops being worth it**, and whether the difference is measured well enough
to say.

---

## The result

| labelled examples | LoRA on Qwen3-1.7B | roberta-base | gap |
|---:|---:|---:|---:|
| 154 | **66.7%** | 56.5% | **+8.9 pp** *(p ≈ 10⁻¹⁸)* |
| 308 | **74.8%** | 71.0% | +3.8 |
| 616 | 81.2% | 82.3% | −1.1 |
| 1,232 | 86.7% | 87.6% | −0.9 |
| 1,848 | 89.3% | 89.5% | −0.2 |
| 9,387 | 93.6% ±0.26 | 94.0% ±0.23 | −0.4 *(p = 0.12, n.s.)* |

**The crossover is between 308 and 616 examples.** Below it the pretrained
model's knowledge substitutes for labelled data and is worth nine points.
Above it, it is worth nothing measurable — at 16× the parameters and 110×
the time per prediction.

| labelled examples | choose | why |
|---|---|---|
| under ~500 | LoRA on the LLM | +8.9 pp; nothing else closes that gap |
| over ~500 | roberta-base | same accuracy, 3 ms vs 333 ms, runs on CPU |

That is the practical output. Everything below is why it should be
believed.

---

## Fine-tuning works, enormously

The comparison above only means something because the adapter itself is not
in doubt. With the **prompt held constant** at 63 tokens, so the LoRA
adapter is the only difference:

| | free-form | constrained |
|---|---:|---:|
| base Qwen3-1.7B | 1.7% | 31.4% |
| **+ LoRA, full pool** | **93.7%** | **93.9%** |
| adapter effect | **+92.0 pp** | **+62.5 pp** |

**1,954 items fixed against 28 broken**, McNemar *p* below floating-point
resolution. LoRA is not marginal here; it takes a model that cannot produce
a valid label at all and makes it competitive with a purpose-built
classifier.

That is exactly why "and it still does not beat a model 16× smaller"
is worth reporting.

---

## Everything measured has the same shape

Three separate interventions, each large when data is scarce and absent
when it is not:

| intervention | at ~600 examples | at 9,387 examples |
|---|---:|---:|
| pretrained knowledge *(LoRA vs encoder)* | +8.9 pp at 154 | −0.4, n.s. |
| adapter capacity *(rank 64 vs 16)* | +3.35 pp | −0.4, n.s. |
| output-format learning *(free-form vs constrained)* | 16.7 pp at 154 | 0.2 pp |

Prior knowledge, extra parameters, retrieved neighbours — they all supply
the same missing thing, and they all stop mattering once there are enough
labels. **Data is the substitute good.**

---

## Six other things this project measured

**In-context examples destroy accuracy.** Zero-shot 45.8%, five exemplars
22.0%, twenty 3.9–12.3%, seventy-seven 5.7%. Constrained decoding gives the
same 5.7% — disagreeing on one item in 3,080 — so it is not a formatting
artefact. The model collapses onto a handful of high-prior intents and
stops reading the query: over 40 queries spanning many classes it produces
30 distinct answers zero-shot and 5–10 with exemplars.
→ `docs/04_baselines.md`

**A retrieval-augmented LLM is 2,200× slower than the retriever it is
repeating.** Retrieval + a 2B model scores 83.1%; kNN over the same
neighbours scores 82.9% at 0.3 ms/item against 667. Indistinguishable
(*p* = 0.18), and the LLM's answer equals the top-1 neighbour's label 81.4%
of the time. → `docs/09_cost.md`

**The MLP modules do most of LoRA's work.** Adapting attention only —
the configuration most tutorials use — costs **7.8 points**, more than a
4× change in rank in either direction, and more than the entire
learning-rate range tested. → `docs/07_ablations.md`

**Output format takes thousands of examples, not dozens.** Unparseable
free-form output runs 96.1% → 34.9% → 5.0% → 0.5% as training data grows
from 0 to 154 to 616 to 9,387. → `docs/06_lora.md`

**Two very different models fail on the same items.** At full data the LoRA
model and the encoder share nearly identical top confusion pairs, and
almost all are symmetric — errors flow both ways. Asymmetric errors are
what data fixes; symmetric ones are boundaries the labels do not draw.
→ `docs/08_taxonomy.md`

**Eval batch size changes predictions.** Batched and unbatched greedy
decoding disagree, by 0.04–0.17 logits — enough to flip any item whose
top-two margin is smaller, and those are precisely the semantically
ambiguous ones. Batch size is pinned and recorded like a random seed.
→ `docs/03_inference.md`

---

## How it was measured

The evaluation was built before the thing it evaluates. That ordering is
visible in the commit history, not just asserted.

**Splits frozen first** — 3,080 test items at exactly 40 per class, a
separate 616-item dev set so failures could be inspected without touching
test, and five nested class-balanced training rungs. Ten artefacts, SHA-256
manifested, **byte-identical across Windows/sklearn 1.9 and Linux/sklearn
1.8**. Twelve integrity tests. → `docs/01_dataset_design.md`

**Evaluator written before any model ran**, with 31 tests in both
directions: twelve that it accepts right answers looking wrong, ten that it
refuses wrong answers looking right. Three-way verdicts —
`correct`/`wrong_label`/`unparseable` — under two decoding regimes, because
a decoder emits text and something has to decide what it meant.
→ `docs/02_evaluator.md`

**Noise measured before any delta was believed.** Training-seed sd is
0.23 pp (encoder) and 0.26 pp (LoRA) at full data — but **0.79 pp at 616
examples**, and importing the wrong one made an ablation threshold three
times too generous. Noise is a property of a regime, not a project.

**Every comparison is paired.** McNemar on the same items, plus a
seed-level Welch test when comparing procedures rather than runs. Both are
reported; where they disagree, the effect is the size of the noise and is
described that way.

**Everything reported twice** — full test and a 2,655-item subset with no
near-duplicate in training — because 13.8% of test has a same-label twin in
the pool. That convention was decided before any model ran.

---

## Predictions and corrections

Twenty-seven outcomes were written down before the runs that would settle
them. **Four were right.**

The pattern is specific and worth more than the tally: **range estimates
land occasionally; mechanism claims essentially never.** Recency anchoring
in the few-shot collapse — refuted, 1 of 3 seeds. Exemplar-set collapse
capping accuracy at *k*/77 — refuted. Near-duplicate memorisation growing
with training size — it *shrinks*, and appears identically for zero-shot,
which has no training set at all. Rank not mattering — wrong at 616
examples, right at 9,387. Fine-tuning fixing output format within dozens of
examples — it takes thousands.

Four errors were made and corrected mid-project, each recorded in the
relevant doc rather than quietly fixed:

| error | effect | found by |
|---|---|---|
| kNN's *k* selected on **test** | +0.5 pp inflation | being asked "are we sure about these numbers?" |
| kNN given 9,387 examples vs the encoder's 616 | reversed a headline | building the matched-data curve |
| LoRA trained 4 epochs vs the encoder's 15 | turned *p* = 0.036 into *p* = 0.119 | noticing dev was still climbing |
| full-pool seed noise applied at rung 8 | threshold 3× too generous | recomputing it from the ablation runs |

The third is the one to read closely. With mismatched epoch budgets the
headline comparison was **significant**. Matched, it is not. The published
claim would have rested on an asymmetry that favoured our own conclusion
and that no reader could have detected.

**Known limitations are collected in each stage doc**, including one that
was not fixed: the failure taxonomy was supposed to read dev and had to
read test, because the training loop scored dev every epoch and threw the
per-item results away.

---

## What this is not

**One model, one task, one label space.** Nothing here establishes where
the crossover sits for a different model size, or a task with 10 classes
instead of 77. The in-context collapse in particular is a Qwen3-1.7B result
on a 77-way problem — published many-shot results generally show gains, and
this is **not** evidence that many-shot ICL fails in general.

**`deberta-v3-base` could not be trained on this stack at all** — fp16
overflow to NaN, then chance-level accuracy for 1,000 steps where
roberta-base reached 84.4% through identical code. Documented with the full
diagnostic chain rather than swapped out silently. → `docs/05_encoder.md`

**Inference timings are indicative, not a serving benchmark.** Single-item
scoring on a Tesla T4. Batching, quantisation and prefix caching would all
narrow the 110× gap, and all help the larger model more.

---

## Reproducing

Every prediction file is committed, so every number regenerates on a laptop
with no GPU.

```bash
python -m venv .venv && .venv/bin/activate
python -m pip install -r requirements.txt

python -m pytest tests -q                 # 43 tests: splits + evaluator
python src/01_build_dataset.py            # rebuild and verify the splits
python src/summarise_runs.py              # every result, from committed runs
python src/compare_families.py robertabase_rungfull qwen3lora_rungfull_ep8
python src/08_cost_latency.py
```

GPU work ran on Kaggle's free tier — Tesla T4, ~27 GPU-hours total.

```
src/            numbered to match docs/ stages; unnumbered are libraries
                (no 02 - the evaluator is a library, not a stage)
docs/           one document per stage, 00-09
eval/splits/    frozen splits, nested rungs, near-duplicate evidence, manifest
reports/runs/   every prediction, with env manifest and full config
tests/          43 tests
```

## Data

[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) — 13,083
customer-service queries over 77 banking intents, CC-BY-4.0, Casanueva et
al. (2020). Read from the original release rather than the Hub mirror,
which is script-based and no longer loadable under `datasets` 5.
