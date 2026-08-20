"""
Stage 1 - probe the candidate dataset before committing to it.

This runs before any model, any prompt, any GPU. It is a go/no-go on
Banking77, and it produces the four numbers that decide whether the
experimental design in the plan is even measurable:

  1. Split sizes -> the resolution table from 00_sizing.py
  2. Class balance -> whether the small rungs of the scaling curve are
     size-limited or coverage-limited (these are different findings)
  3. Cross-split near-duplicates -> whether test is honest
  4. Same-text-different-label pairs -> the label-noise ceiling. No model
     can beat it, and it caps every accuracy number in the project.

Requires: datasets, scikit-learn, numpy, pandas. Internet on first run
(it caches to ~/.cache/huggingface afterwards).

    python 01_dataset_probe.py
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATASET = "PolyAI/banking77"
NEAR_DUP_COSINE = 0.90


def rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def load():
    ds = load_dataset(DATASET)
    names = ds["train"].features["label"].names
    frames = {}
    for split in ds:
        frames[split] = pd.DataFrame(
            {"text": ds[split]["text"],
             "label": [names[i] for i in ds[split]["label"]]}
        )
    return frames, names


def sizes(frames, names):
    rule("1. Split sizes")
    for split, df in frames.items():
        print(f"  {split:6s} {len(df):6,d} rows   {df.label.nunique():3d} classes")
    print(f"  {len(names)} intents total")
    n = len(frames["test"])
    half = 1.96 * np.sqrt(0.85 * 0.15 / n) * 100
    print(f"\n  95% CI half-width on a test accuracy near 85%: +/-{half:.1f} pp")
    for psi in (0.05, 0.10, 0.20):
        mdd = 2.80 * np.sqrt(psi / n) * 100
        print(f"  paired MDD at {psi:.0%} discordance, 80% power: {mdd:.1f} pp")


def balance(frames):
    rule("2. Class balance in the training pool")
    c = Counter(frames["train"].label)
    counts = np.array(sorted(c.values()))
    print(f"  per-class count  min {counts.min()}  median "
          f"{int(np.median(counts))}  max {counts.max()}")
    print(f"  imbalance ratio  {counts.max() / counts.min():.2f}x")

    print("\n  scaling-curve rungs, two ways of drawing them:")
    print("    flat sample      per-class     starved classes (<2 examples)")
    n_cls = len(c)
    for n_ex in (200, 400, 800, 1600):
        per = n_ex / n_cls
        starved = sum(1 for _ in range(n_cls) if per < 2)
        print(f"    {n_ex:5d} examples  {per:6.1f}      {starved:3d} of {n_cls}")
    print("\n  per-class rungs (recommended - removes the coverage confound):")
    for k in (2, 4, 8, 16, 32):
        total = sum(min(k, v) for v in c.values())
        print(f"    {k:2d} per class -> {total:5d} examples")


def duplicates(frames):
    rule("3. Duplicates and cross-split leakage")
    tr, te = frames["train"], frames["test"]
    tr_norm, te_norm = tr.text.map(norm), te.text.map(norm)

    print(f"  exact dups within train: "
          f"{len(tr_norm) - tr_norm.nunique():,d}")
    print(f"  exact dups within test:  "
          f"{len(te_norm) - te_norm.nunique():,d}")
    overlap = set(tr_norm) & set(te_norm)
    print(f"  exact train/test overlap: {len(overlap):,d} unique strings "
          f"({len(overlap) / te_norm.nunique() * 100:.1f}% of test)")

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X_tr = vec.fit_transform(tr.text)
    X_te = vec.transform(te.text)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X_tr)
    dist, idx = nn.kneighbors(X_te)
    sim = 1 - dist.ravel()
    near = sim >= NEAR_DUP_COSINE
    print(f"  near-dups (cosine >= {NEAR_DUP_COSINE}): {near.sum():,d} "
          f"({near.mean() * 100:.1f}% of test)")

    same_label = (tr.label.values[idx.ravel()] == te.label.values)
    both = near & ~same_label
    print(f"  of those, labelled differently from their train twin: "
          f"{both.sum():,d}")
    if both.sum():
        print("\n  examples (these cap achievable accuracy):")
        for i in np.where(both)[0][:5]:
            j = idx.ravel()[i]
            print(f"    sim {sim[i]:.2f}")
            print(f"      test  [{te.label.values[i]}] {te.text.values[i][:70]}")
            print(f"      train [{tr.label.values[j]}] {tr.text.values[j][:70]}")


def confusability(frames):
    rule("4. Within-test confusable pairs (label-noise ceiling)")
    te = frames["test"]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X = vec.fit_transform(te.text)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    sim = 1 - dist[:, 1]
    peer = idx[:, 1]
    disagree = (te.label.values[peer] != te.label.values) & (sim >= 0.85)
    print(f"  test items whose nearest neighbour is >=0.85 similar but "
          f"differently labelled: {disagree.sum():,d} "
          f"({disagree.mean() * 100:.1f}%)")
    print("\n  most confused intent pairs:")
    pairs = Counter()
    for i in np.where(disagree)[0]:
        a, b = sorted([te.label.values[i], te.label.values[peer[i]]])
        pairs[(a, b)] += 1
    for (a, b), n in pairs.most_common(8):
        print(f"    {n:3d}  {a}  <->  {b}")


def lengths(frames):
    rule("5. Text length (drives max_seq_len and therefore GPU time)")
    for split, df in frames.items():
        w = df.text.str.split().str.len()
        print(f"  {split:6s} words: p50 {w.quantile(.5):.0f}  "
              f"p95 {w.quantile(.95):.0f}  max {w.max()}")


if __name__ == "__main__":
    frames, names = load()
    sizes(frames, names)
    balance(frames)
    duplicates(frames)
    confusability(frames)
    lengths(frames)
    print("\nNo split has been written yet. This script only looks.\n")
