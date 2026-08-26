"""
Stage 1 - probe the candidate dataset before committing to it.

This runs before any model, any prompt, any GPU. It is a go/no-go on
Banking77, and it produces the four numbers that decide whether the
experimental design in the plan is even measurable:

  1. Split sizes -> the resolution table from 00a_sizing.py
  2. Class balance -> whether the small rungs of the scaling curve are
     size-limited or coverage-limited (these are different findings)
  3. Cross-split near-duplicates -> whether test is honest
  4. Same-text-different-label pairs -> the label-noise ceiling. No model
     can beat it, and it caps every accuracy number in the project.

Provenance note. Banking77 is on the Hub as PolyAI/banking77, but that repo
is still script-based and datasets>=5 dropped script loading; the parquet
conversion is an unmerged PR, so its revision can move. We read the original
PolyAI release from GitHub instead - CC-BY-4.0, plain CSV, no coupling to a
`datasets` version and no dependence on a PR that may be rebased. Files are
cached under data/raw/ on first run.

    python 01a_probe_dataset.py
"""

import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

BASE = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
        "/master/banking_data")
CACHE = Path("data/raw")
NEAR_DUP_COSINE = 0.90
EXPECTED = {"train": 10_003, "test": 3_080, "classes": 77}


def rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in ("train", "test"):
        local = CACHE / f"banking77_{split}.csv"
        if not local.exists():
            print(f"  downloading {split} ...")
            urllib.request.urlretrieve(f"{BASE}/{split}.csv", local)
        df = pd.read_csv(local).rename(columns={"category": "label"})
        frames[split] = df[["text", "label"]].reset_index(drop=True)

    # Fail loudly if upstream ever changes shape - a silent reshape here
    # would invalidate every downstream number without any error.
    got = {"train": len(frames["train"]), "test": len(frames["test"]),
           "classes": frames["train"].label.nunique()}
    if got != EXPECTED:
        raise RuntimeError(f"upstream shape changed: expected {EXPECTED}, got {got}")
    if set(frames["train"].label) != set(frames["test"].label):
        raise RuntimeError("train and test label sets differ")

    names = sorted(frames["train"].label.unique())
    print("loaded banking77 from upstream CSV: "
          + ", ".join(f"{k} {len(v):,d}" for k, v in frames.items()))
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
    print("  smallest classes: " + ", ".join(
        f"{k}({v})" for k, v in sorted(c.items(), key=lambda kv: kv[1])[:5]))

    n_cls = len(c)
    print("\n  flat rungs: examples per class if sampled without stratifying")
    for n_ex in (200, 400, 800, 1600):
        print(f"    {n_ex:5d} examples -> {n_ex / n_cls:5.1f} per class")
    print("\n  per-class rungs (recommended - removes the coverage confound):")
    for k in (2, 4, 8, 16, 32):
        total = sum(min(k, v) for v in c.values())
        short = sum(1 for v in c.values() if v < k)
        print(f"    {k:2d} per class -> {total:5d} examples ({short} classes short)")


def duplicates(frames):
    rule("3. Duplicates and cross-split leakage")
    tr, te = frames["train"], frames["test"]
    tr_norm, te_norm = tr.text.map(norm), te.text.map(norm)

    print(f"  exact dups within train: {len(tr_norm) - tr_norm.nunique():,d}")
    print(f"  exact dups within test:  {len(te_norm) - te_norm.nunique():,d}")
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

    same_label = tr.label.values[idx.ravel()] == te.label.values
    both = near & ~same_label
    print(f"  of those, labelled differently from their train twin: {both.sum():,d}")
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
    print("\n  sample pairs:")
    shown = 0
    for i in np.where(disagree)[0]:
        if shown >= 4:
            break
        print(f"    sim {sim[i]:.2f}")
        print(f"      [{te.label.values[i]}] {te.text.values[i][:70]}")
        print(f"      [{te.label.values[peer[i]]}] {te.text.values[peer[i]][:70]}")
        shown += 1


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
