# Stage 1 — Dataset and splits

Written before any model was run. Every decision here is frozen; changing
one invalidates every number produced after it.

## Source and provenance

Banking77 — 13,083 customer-service queries over 77 fine-grained banking
intents, CC-BY-4.0, from PolyAI (Casanueva et al., 2020).

Read from the **original GitHub release**, not the Hugging Face mirror. The
Hub repo `PolyAI/banking77` is still script-based, and `datasets>=5` removed
script loading; the parquet conversion exists only as an unmerged pull
request whose revision can be rebased or closed. The upstream CSVs are plain,
stable, and carry no dependency on a `datasets` version — which matters
because training happens on Kaggle, whose image pins its own.

`load_raw()` asserts the exact shape (10,003 / 3,080 / 77) on every run. A
silent upstream reshape would invalidate everything downstream without
raising anything, so it raises.

## Splits

| Split | Rows | Per class | Purpose |
|---|---:|---:|---|
| `test` | 3,080 | 40 | Frozen. Touched once per completed config. |
| `dev` | 616 | 8 | Error taxonomy, prompt iteration, anything that looks at failures. |
| `train_pool` | 9,387 | 27–179 | Source of every training subset. |

**Test is the official split, unmodified.** It is exactly 40 per class, so
accuracy equals macro-accuracy and there is no class-prior shift to reason
about. Not deduplicated: removing items would break comparability with every
published Banking77 number.

**The three-way split is the main methodological change from project 2.**
There, the intervention was a prompt and there was nothing to overfit with.
Here the training set *is* the intervention, so the temptation on seeing a
failure mode is to add training examples covering it — which is fitting to
the test set with extra steps. Failures are read on `dev`. `test` is scored
once per frozen config and never inspected.

**IDs come from upstream row order** (`train-00042`, `test-00017`), so they
are stable across machines with no mapping file to keep in sync.

## Scaling-curve rungs

2 / 4 / 8 / 16 / 24 examples per class → 154 / 308 / 616 / 1,232 / 1,848.

**Per class, not flat.** A flat 200-example sample over 77 classes gives 2.6
per class and leaves some classes unrepresented, which confounds *dataset
size* with *class coverage*. Those are different findings and a curve that
mixes them can't distinguish "more data helps" from "seeing the class at all
helps".

**Nested.** One permutation per class, fixed seed, rungs are prefixes. Each
rung is a strict superset of the one below (asserted in code). Independently
drawn rungs would add resampling noise between adjacent points on a curve
whose adjacent points we already expect to be hard to separate.

**Ceiling of 24, not 32.** The smallest class has 35 examples; after the dev
carve the pool's smallest class holds 27. 24 is the largest balanced rung
with no starved class.

## Near-duplicate evidence

For every `test` and `dev` row, the nearest neighbour in `train_pool` is
recorded as raw fields — `twin_id`, `twin_label`, `twin_sim` (char 3–5-gram
TF-IDF, cosine) — in a **separate file**, `near_dup_{test,dev}.jsonl`, joined
on `id`.

**Evidence, not a flag.** A later question about a different threshold is a
filter, not a recomputation, and no threshold choice is baked into the frozen
artefact.

**Separate file, not extra columns.** The manifest distinguishes
`sha256_core` (identity — which items, which labels) from `sha256_derived`
(analysis output). TF-IDF cannot see semantic duplicates, so the
near-duplicate method may well be improved later; under a single-file design
that improvement would change `test.jsonl`'s hash and invalidate every
earlier run's manifest reference, despite the test set never having moved.
Identity is immutable. Analysis is versioned.

At cosine ≥ 0.90:

| | Near-dup | Same label | Different label | Clean subset |
|---|---:|---:|---:|---:|
| test | 425 (13.8%) | 400 | 25 | 2,655 |
| dev | 79 (12.8%) | 78 | 1 | 537 |

**Why this changes the reporting.** ~13% of test has a same-label near-twin
in the pool. A fine-tuned model can recall those; zero-shot and few-shot
prompting cannot. So part of any "LoRA beats few-shot" gap is memorisation
rather than generalisation, and it grows with dataset size — biasing the
scaling curve in exactly the direction that flatters fine-tuning.

**Every accuracy in this project is therefore reported twice:** on the full
3,080, and on the 2,655-item clean subset. The gap between them measures how
much of the gain is recall. This was decided before any model ran; deciding
it afterwards would not be defensible.

One known imperfection: the clean subset is fixed across all configs, while
the actual memorisation opportunity varies by rung — at 2 per class, few of
the 425 twins are even in the training subset. The stored `twin_id` makes
per-rung twin membership computable at analysis time, and that correction
will be reported alongside the fixed-subset one rather than replacing it.

## What this measurement cannot see

TF-IDF character n-grams catch lexical similarity only. Two differently
worded requests for the same thing are invisible to it. The 22 within-test
confusable pairs (0.7%) found by the probe are therefore a **floor** on
irreducible label noise, not an estimate of it. Published Banking77 results
sit near 93–94%; the gap between that and 99.3% is semantic confusion this
method cannot detect.

## Resolution this buys

With n = 3,080, the 95% CI half-width on a single accuracy near 85% is
±1.3 pp, and the paired minimum detectable difference at 10% discordance is
1.6 pp. The test set is not the binding constraint on this project —
training-seed variance is, and it is measured in stage 5 before any ablation
is believed.

## Written predictions

Recorded before training, so a miss is itself a result.

1. The largest failure cluster will be the intent pairs already visible in
   the probe — `verify_my_identity`/`why_verify_identity`,
   `card_arrival`/`card_delivery_estimate`, `exchange_charge`/`exchange_rate`
   — and they will persist at every rung, because more examples of a
   genuinely ambiguous boundary do not resolve it.
2. Full-test accuracy will exceed clean-subset accuracy for every fine-tuned
   config, and the gap will widen with rung size.
3. Few-shot and zero-shot baselines will show no meaningful full-vs-clean
   gap, since they have no training set to recall from.

## Reproducibility

The splits were built independently on Windows/scikit-learn 1.9 and
Linux/scikit-learn 1.8. Reconciling the two exposed two defects, both fixed
in code rather than documented as quirks.

**Line-ending translation.** `Path.write_text()` defaults to platform
newlines, so the rung files and manifest were CRLF on Windows and LF on
Linux — different bytes, different hashes, identical content. All writes now
force `newline="\n"`. A `.gitattributes` marks the hashed artefacts `-text`
so Git's `autocrlf` cannot rewrite them on checkout either.

**Non-deterministic tie-breaking.** Duplicate texts in `train_pool` produce
identical TF-IDF vectors and therefore exact ties for nearest neighbour;
`NearestNeighbors` resolved them by internal ordering, which is not stable
across scikit-learn versions. Six test rows were affected. Similarities and
twin *labels* were bit-identical — only the twin *id* differed, so no
reported count ever moved. The lookup now retrieves 8 candidates, collects
everything within 1e-12 of the best similarity, and selects the smallest
`id`; it raises if a tie group saturates the candidate list rather than
silently truncating it.

Neither defect would have changed a headline number. Both would have made
the integrity check fire for reasons unrelated to the data — and an integrity
check that cries wolf is worse than none, because it trains you to ignore it.

After the fixes, all ten artefact hashes agree across both machines.

## Artefacts

```
eval/splits/test.jsonl            3,080  id, text, label            [core]
eval/splits/dev.jsonl               616  id, text, label            [core]
eval/splits/train_pool.jsonl      9,387  id, text, label            [core]
eval/splits/rungs/rung_NN.json           nested ID lists            [core]
eval/splits/near_dup_test.jsonl   3,080  id, twin_id, twin_label, twin_sim  [derived]
eval/splits/near_dup_dev.jsonl      616  same fields                [derived]
eval/splits/manifest.json                counts, seed, params, scoped SHA-256
```

Committed. `tests/test_dataset.py` re-verifies hashes, nesting, class
balance, dev/pool disjointness, and the clean-subset denominator on every
run — 12 assertions.
