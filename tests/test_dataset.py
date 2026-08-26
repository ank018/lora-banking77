"""Split integrity.

These do not test that the splits are *good* - stage 1's doc argues that.
They test that the splits have not moved. Every number in this project is
reported against these files, so a silent change to any of them silently
invalidates the project.

    pytest tests/test_dataset.py -v
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

SPLITS = Path("eval/splits")
MANIFEST = SPLITS / "manifest.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip("splits not built - run src/01b_build_dataset.py")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def splits():
    return {name: read_jsonl(SPLITS / f"{name}.jsonl")
            for name in ("test", "dev", "train_pool")}


# --------------------------------------------------------------- identity
def test_core_hashes_unchanged(manifest):
    """The identity of the splits. Bit-identical on every machine."""
    mismatched = []
    for path, expected in manifest["sha256_core"].items():
        actual = sha256_file(path)
        if actual != expected:
            mismatched.append(f"{path}: {expected[:16]} -> {actual[:16]}")
    assert not mismatched, "core artefacts changed:\n  " + "\n  ".join(mismatched)


def test_derived_hashes_unchanged(manifest):
    """Analysis output. Recomputable; a change here is a method change,
    not a data change, and does not invalidate the splits themselves."""
    for path, expected in manifest["sha256_derived"].items():
        assert sha256_file(path) == expected, f"{path} changed"


def test_counts_match_manifest(manifest, splits):
    for name, rows in splits.items():
        assert len(rows) == manifest["counts"][name]


# ------------------------------------------------------------ no leakage
def test_dev_and_pool_are_disjoint(splits):
    dev = {r["id"] for r in splits["dev"]}
    pool = {r["id"] for r in splits["train_pool"]}
    assert not (dev & pool)


def test_ids_are_unique_within_each_split(splits):
    for name, rows in splits.items():
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate ids in {name}"


def test_test_ids_never_appear_in_training_artefacts(splits, manifest):
    test_ids = {r["id"] for r in splits["test"]}
    pool_ids = {r["id"] for r in splits["train_pool"]}
    assert not (test_ids & pool_ids)
    for k in manifest["rungs_per_class"]:
        rung = set(json.loads((SPLITS / "rungs" / f"rung_{k:02d}.json")
                              .read_text(encoding="utf-8")))
        assert not (test_ids & rung), f"test ids leaked into rung {k}"
        assert rung <= pool_ids, f"rung {k} contains ids outside train_pool"


# ----------------------------------------------------------------- shape
def test_test_is_balanced_at_40_per_class(splits):
    counts = Counter(r["label"] for r in splits["test"])
    assert set(counts.values()) == {40}
    assert len(counts) == 77


def test_dev_is_balanced_at_configured_size(manifest, splits):
    counts = Counter(r["label"] for r in splits["dev"])
    assert set(counts.values()) == {manifest["dev_per_class"]}


def test_rungs_are_nested_and_balanced(manifest):
    rungs = manifest["rungs_per_class"]
    loaded = {k: json.loads((SPLITS / "rungs" / f"rung_{k:02d}.json")
                            .read_text(encoding="utf-8")) for k in rungs}
    pool_label = {r["id"]: r["label"] for r in read_jsonl(SPLITS / "train_pool.jsonl")}
    for k in rungs:
        counts = Counter(pool_label[i] for i in loaded[k])
        assert set(counts.values()) == {k}, f"rung {k} is not balanced"
        assert len(counts) == 77
    for a, b in zip(rungs, rungs[1:]):
        assert set(loaded[a]) <= set(loaded[b]), f"rung {a} not nested in {b}"


# -------------------------------------------------------------- evidence
@pytest.mark.parametrize("name", ["test", "dev"])
def test_near_dup_evidence_aligns_with_split(name, splits):
    ev = read_jsonl(SPLITS / f"near_dup_{name}.jsonl")
    assert [r["id"] for r in ev] == [r["id"] for r in splits[name]]
    pool_ids = {r["id"] for r in splits["train_pool"]}
    assert all(r["twin_id"] in pool_ids for r in ev)
    assert all(0.0 <= r["twin_sim"] <= 1.0 for r in ev)


def test_clean_subset_size_is_stable(manifest, splits):
    """The denominator every clean-subset accuracy is reported against."""
    ev = read_jsonl(SPLITS / "near_dup_test.jsonl")
    thr = manifest["near_dup_threshold_recorded"]
    clean = sum(1 for r in ev if r["twin_sim"] < thr)
    assert clean == 2655, f"clean subset moved to {clean}"
