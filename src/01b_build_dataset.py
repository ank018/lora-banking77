"""
Stage 1 - build and freeze the splits.

Produces every artefact the rest of the project is measured against, and
nothing else in the repo may create a split. Written once, committed, and
hashed; if these files change, every number downstream is invalidated and
tests/test_dataset.py will say so.

Design decisions are argued in docs/01_dataset_design.md. In brief:

  test        official 3,080, untouched, exactly 40 per class
  dev         8 per class (616) carved from official train, stratified
  train_pool  the remaining 9,387
  rungs       2/4/8/16/24 per class, NESTED - each rung is a superset of
              the one below, so movement along the curve is caused by the
              added examples and not by resampling

Near-duplicate evidence is stored per test item as raw fields (twin id,
similarity, twin label) rather than as a boolean. Flags are derived at
analysis time; storing the evidence means a later question about a
different threshold does not require recomputing anything.

    python src/01b_build_dataset.py
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

RAW = Path("data/raw")
OUT = Path("eval/splits")
SEED = 20260821
DEV_PER_CLASS = 8
RUNGS = [2, 4, 8, 16, 24]
NEAR_DUP_THRESHOLD = 0.90  # recorded, not applied - flags derive from sim
EXPECTED = {"train": 10_003, "test": 3_080, "classes": 77}


def rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw():
    frames = {}
    for split in ("train", "test"):
        p = RAW / f"banking77_{split}.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing - run src/01a_probe_dataset.py first")
        df = pd.read_csv(p).rename(columns={"category": "label"})
        df = df[["text", "label"]].reset_index(drop=True)
        # IDs come from upstream row order, so they are stable across
        # machines and across reruns without storing an extra mapping.
        df["id"] = [f"{split}-{i:05d}" for i in range(len(df))]
        frames[split] = df
    got = {"train": len(frames["train"]), "test": len(frames["test"]),
           "classes": frames["train"].label.nunique()}
    if got != EXPECTED:
        raise RuntimeError(f"upstream shape changed: expected {EXPECTED}, got {got}")
    return frames


def carve_dev(train):
    """Stratified dev split: exactly DEV_PER_CLASS per class, fixed seed."""
    rng = np.random.default_rng(SEED)
    dev_ids = []
    for label, grp in train.groupby("label", sort=True):
        ids = grp.id.to_numpy()
        rng.shuffle(ids)
        dev_ids.extend(ids[:DEV_PER_CLASS].tolist())
    dev_ids = set(dev_ids)
    dev = train[train.id.isin(dev_ids)].reset_index(drop=True)
    pool = train[~train.id.isin(dev_ids)].reset_index(drop=True)
    return dev, pool


def build_rungs(pool):
    """Nested class-balanced subsets. One permutation per class, then prefixes."""
    rng = np.random.default_rng(SEED + 1)
    order = {}
    for label, grp in pool.groupby("label", sort=True):
        ids = grp.id.to_numpy()
        rng.shuffle(ids)
        order[label] = ids.tolist()

    smallest = min(len(v) for v in order.values())
    if max(RUNGS) > smallest:
        raise RuntimeError(f"rung {max(RUNGS)} exceeds smallest class "
                           f"in pool ({smallest})")

    rungs = {}
    for k in RUNGS:
        ids = [i for label in sorted(order) for i in order[label][:k]]
        rungs[k] = sorted(ids)
    # Nesting is the whole point of the construction; assert it anyway.
    for a, b in zip(RUNGS, RUNGS[1:]):
        assert set(rungs[a]).issubset(rungs[b]), f"rung {a} not nested in {b}"
    return rungs


def nearest_in_pool(target, pool, k=8):
    """Nearest neighbour in pool by char-ngram tfidf cosine.

    Duplicate texts in the pool produce identical vectors and therefore
    exact ties, which sklearn breaks by internal ordering - that ordering
    is not stable across versions. We retrieve k candidates and break ties
    ourselves on the smallest id (zero-padded, so lexicographic order is
    numeric order). Deterministic on any machine, any sklearn build.
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X_pool = vec.fit_transform(pool.text)
    X_tgt = vec.transform(target.text)
    k = min(k, len(pool))
    dist, idx = NearestNeighbors(n_neighbors=k, metric="cosine").fit(
        X_pool).kneighbors(X_tgt)
    sims = 1 - dist

    pool_ids, pool_labels = pool.id.to_numpy(), pool.label.to_numpy()
    twin_id, twin_label, twin_sim = [], [], []
    n_tied, n_saturated = 0, 0
    for row_sim, row_idx in zip(sims, idx):
        best = row_sim.max()
        tied = row_sim >= best - 1e-12
        if tied.sum() > 1:
            n_tied += 1
        if tied.sum() == k:
            n_saturated += 1
        cands = row_idx[tied]
        j = cands[pool_ids[cands].argmin()]
        twin_id.append(pool_ids[j])
        twin_label.append(pool_labels[j])
        twin_sim.append(round(float(best), 4))

    if n_saturated:
        raise RuntimeError(f"{n_saturated} rows tied across all k={k} "
                           f"candidates - raise k, the tie-break is truncated")

    evidence = pd.DataFrame({
        "id": target.id.to_numpy(),
        "twin_id": twin_id,
        "twin_label": twin_label,
        "twin_sim": twin_sim,
    })
    return evidence, {"tied_rows": n_tied}


def write_jsonl(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def main():
    frames = load_raw()
    train, test = frames["train"], frames["test"]

    rule("splits")
    dev, pool = carve_dev(train)
    print(f"  test        {len(test):6,d}  ({Counter(test.label).most_common(1)[0][1]} per class)")
    print(f"  dev         {len(dev):6,d}  ({DEV_PER_CLASS} per class)")
    print(f"  train_pool  {len(pool):6,d}  (min class "
          f"{min(Counter(pool.label).values())})")
    assert not (set(dev.id) & set(pool.id)), "dev/pool overlap"

    rule("rungs (nested, class-balanced)")
    rungs = build_rungs(pool)
    for k in RUNGS:
        print(f"  {k:2d} per class -> {len(rungs[k]):5,d} examples")

    rule("near-duplicate evidence vs train_pool")
    evidence = {}
    for name, df in (("test", test), ("dev", dev)):
        ev, stats = nearest_in_pool(df, pool)
        evidence[name] = ev
        merged = df.merge(ev, on="id")
        near = merged.twin_sim >= NEAR_DUP_THRESHOLD
        same = merged.twin_label == merged.label
        print(f"  {name:5s} near-dup {near.sum():5,d} "
              f"({near.mean() * 100:4.1f}%)   "
              f"same-label {int((near & same).sum()):5,d}   "
              f"diff-label {int((near & ~same).sum()):4,d}   "
              f"clean subset n={int((~near).sum()):5,d}   "
              f"[{stats['tied_rows']} tied rows resolved by id]")

    rule("written")
    core, derived = {}, {}
    for name, df in (("test", test), ("dev", dev), ("train_pool", pool)):
        p = write_jsonl(df[["id", "text", "label"]], OUT / f"{name}.jsonl")
        core[str(p).replace("\\", "/")] = sha256_file(p)
        print(f"  {p}  {len(df):,d} rows")
    for k in RUNGS:
        p = OUT / "rungs" / f"rung_{k:02d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" is load-bearing: the default translates to CRLF on
        # Windows, which changes the bytes and therefore the hash, so the
        # same splits would fail the integrity check between a Windows box
        # and a Linux GPU host.
        p.write_text(json.dumps(rungs[k], indent=0), encoding="utf-8",
                     newline="\n")
        core[str(p).replace("\\", "/")] = sha256_file(p)
    for name, ev in evidence.items():
        p = write_jsonl(ev, OUT / f"near_dup_{name}.jsonl")
        derived[str(p).replace("\\", "/")] = sha256_file(p)
        print(f"  {p}  {len(ev):,d} rows")

    manifest = {
        "source": "PolyAI-LDN/task-specific-datasets banking_data (CC-BY-4.0)",
        "seed": SEED,
        "dev_per_class": DEV_PER_CLASS,
        "rungs_per_class": RUNGS,
        "near_dup_threshold_recorded": NEAR_DUP_THRESHOLD,
        "counts": {"test": len(test), "dev": len(dev), "train_pool": len(pool),
                   **{f"rung_{k:02d}": len(rungs[k]) for k in RUNGS}},
        # Identity. Must be bit-identical everywhere; tests assert it.
        "sha256_core": core,
        # Analysis output. Recomputable, and revisable if the near-duplicate
        # method changes, without altering the identity of the splits.
        "sha256_derived": derived,
    }
    mp = OUT / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8",
                  newline="\n")
    print(f"  {mp}")

    # Per-file hashes, so a mismatch names the offending file instead of
    # only telling us that something, somewhere, differs.
    print("\n  sha256 (first 16)")
    for scope, group in (("core", core), ("derived", derived)):
        for path, digest in sorted(group.items()):
            print(f"    {digest[:16]}  [{scope:7s}] {path}")
    print(f"\n  manifest sha256: {sha256_file(mp)[:16]}")
    print("\nSplits are frozen. Nothing else in this repo may create a split.\n")


if __name__ == "__main__":
    main()
