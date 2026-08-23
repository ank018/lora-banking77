"""
The baseline with no model in it.

retrieval_k10 scored 82.7% against zero-shot's 45.8%. The only difference
between it and few_shot_k5 - which scored 22.0% - is whether the exemplars
are relevant to the query. That pattern is what you would see if the model
were copying its nearest neighbour's label rather than classifying.

If so, the retrieval step is doing the work and the 2-billion-parameter
model on top of it is decoration. The way to find out is to run the
retrieval step alone: nearest neighbour over the training pool by the same
TF-IDF features, no language model anywhere.

Two views:

  full pool   all 9,387 training examples as the reference set
  per rung    restricted to the same 154-1,848 examples the encoder was
              trained on, so both methods share an x-axis

The second matters. Comparing full-pool kNN against an encoder trained on
616 examples flatters the encoder by a factor of fifteen on data, and the
first version of this script did exactly that.

    python src/03c_knn_baseline.py --rungs
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from evaluator import evaluate_constrained, score  # noqa: E402

SPLITS = Path("eval/splits")
RUNS = Path("reports/runs")
NEIGHBOURS = [1, 3, 5, 10, 20]
RUNGS = [2, 4, 8, 16, 24]

# k is chosen on dev, never on test. The first version of this script ran
# every k on test and quoted the best - that is tuning on the test set, and
# it inflated the headline by 0.5 points. Test scores for all k are still
# printed, because hiding them would be its own kind of dishonesty, but the
# reported figure is the k that dev picked.


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_ids():
    m = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    thr = m["near_dup_threshold_recorded"]
    return [r["id"] for r in read_jsonl(SPLITS / "near_dup_test.jsonl")
            if r["twin_sim"] < thr]


def predict(pool, target, k):
    """k-NN over char n-gram TF-IDF, similarity-weighted vote.

    Ties are broken by nearer neighbour then by pool id, so the result does
    not depend on sklearn's internal ordering - the same determinism issue
    that produced a hash mismatch in stage 1.
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform([r["text"] for r in pool])
    Q = vec.transform([r["text"] for r in target])
    k = min(k, len(pool))
    dist, idx = NearestNeighbors(n_neighbors=k, metric="cosine").fit(
        X).kneighbors(Q)

    preds, top1_labels = [], []
    for row_d, row_i in zip(dist, idx):
        order = sorted(zip(1 - row_d, row_i),
                       key=lambda p: (-p[0], pool[p[1]]["id"]))
        top1_labels.append(pool[order[0][1]]["label"])
        if k == 1:
            preds.append(top1_labels[-1])
            continue
        weight = Counter()
        for sim, j in order:
            weight[pool[j]["label"]] += sim
        preds.append(max(sorted(weight), key=lambda lbl: weight[lbl]))
    return preds, top1_labels


def write_run(runs, name, recs, meta):
    d = runs / f"{name}__constrained"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "predictions.jsonl", "w", encoding="utf-8",
              newline="\n") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    (d / "meta.json").write_text(json.dumps(meta, indent=2),
                                 encoding="utf-8", newline="\n")


def select_k_on_dev(pool, dev, candidates):
    acc = {}
    for k in candidates:
        preds, _ = predict(pool, dev, k)
        acc[k] = sum(p == r["label"] for p, r in zip(preds, dev)) / len(dev)
    return max(sorted(acc), key=lambda k: acc[k]), acc


def full_pool(pool, dev, test, labels, clean, runs):
    print(f"\n  selecting k on dev ({len(dev):,d} items), test untouched")
    best_k, dev_acc = select_k_on_dev(pool, dev, NEIGHBOURS)
    for k in NEIGHBOURS:
        print(f"    k={k:<3d} dev {dev_acc[k] * 100:5.1f}%")
    print(f"    -> k={best_k} selected on dev")

    print(f"\n  {'config':12s} {'acc':>8s} {'clean':>8s}   (no model, no GPU)")
    results = {}
    for k in NEIGHBOURS:
        preds, _ = predict(pool, test, k)
        recs = [evaluate_constrained(p, r["label"], labels)
                | {"id": r["id"], "query": r["text"]}
                for p, r in zip(preds, test)]
        write_run(runs, f"knn_k{k}", recs, {
            "config": f"knn_k{k}", "regime": "constrained",
            "model": "tfidf char_wb 3-5 + kNN", "k": k,
            "n_pool": len(pool), "n_items": len(test)})
        s, c = score(recs), score(recs, subset_ids=clean)
        results[k] = (s, c)
        mark = "  <- selected on dev" if k == best_k else ""
        print(f"  knn_k{k:<7d} {s['accuracy'] * 100:7.1f}% "
              f"{c['accuracy'] * 100:7.1f}%{mark}")

    s = results[best_k][0]
    half = 1.96 * math.sqrt(s["accuracy"] * (1 - s["accuracy"]) / s["n"]) * 100
    best_on_test = max(r[0]["accuracy"] for r in results.values()) * 100
    worst_on_test = min(r[0]["accuracy"] for r in results.values()) * 100
    print(f"\n  REPORTED: knn_k{best_k}  {s['accuracy'] * 100:.1f}% "
          f"+/- {half:.1f} pp (95% CI, n={s['n']:,d})")
    print(f"  best-of-{len(NEIGHBOURS)} on test would have been "
          f"{best_on_test:.1f}% (spread across k: "
          f"{best_on_test - worst_on_test:.1f} pp)")


def rung_curve(pool, dev, test, labels, clean, runs):
    """kNN restricted to each training rung.

    The headline kNN figure uses all 9,387 pool examples while the encoder
    sees 154-1,848. Restricting kNN's reference set to the same rungs puts
    both methods on one x-axis, which is the only version of this
    comparison worth reporting.
    """
    print(f"\n{'=' * 68}")
    print("kNN restricted to each rung (same data the encoder saw)")
    print(f"{'=' * 68}")
    print(f"  {'per class':>9s} {'n_pool':>8s} {'k(dev)':>7s} "
          f"{'test':>8s} {'clean':>8s}")

    by_id = {r["id"]: r for r in pool}
    for rung in RUNGS:
        ids = json.loads((SPLITS / "rungs" / f"rung_{rung:02d}.json")
                         .read_text(encoding="utf-8"))
        sub = [by_id[i] for i in ids]
        cands = [n for n in NEIGHBOURS if n <= len(sub)]
        best_k, _ = select_k_on_dev(sub, dev, cands)

        preds, _ = predict(sub, test, best_k)
        recs = [evaluate_constrained(p, r["label"], labels)
                | {"id": r["id"], "query": r["text"]}
                for p, r in zip(preds, test)]
        write_run(runs, f"knn_rung{rung:02d}", recs, {
            "config": f"knn_rung{rung:02d}", "regime": "constrained",
            "model": "tfidf char_wb 3-5 + kNN", "rung_per_class": rung,
            "n_pool": len(sub), "k_selected_on_dev": best_k,
            "n_items": len(test)})
        s, c = score(recs), score(recs, subset_ids=clean)
        print(f"  {rung:9d} {len(sub):8,d} {best_k:7d} "
              f"{s['accuracy'] * 100:7.1f}% {c['accuracy'] * 100:7.1f}%")

    print("\n  Compare against the roberta curve at the same rungs. Where")
    print("  they cross is the amount of labelled data at which training a")
    print("  model starts to beat looking the answer up.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--rungs", action="store_true",
                    help="also run kNN restricted to each training rung")
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    pool = read_jsonl(SPLITS / "train_pool.jsonl")
    test = read_jsonl(SPLITS / "test.jsonl")
    dev = read_jsonl(SPLITS / "dev.jsonl")
    labels = sorted({r["label"] for r in pool})
    clean = clean_ids()
    print(f"pool {len(pool):,d}   test {len(test):,d}   dev {len(dev):,d}   "
          f"clean {len(clean):,d}   labels {len(labels)}")

    full_pool(pool, dev, test, labels, clean, runs)
    if args.rungs:
        rung_curve(pool, dev, test, labels, clean, runs)

    rk = runs / "retrieval_k10__free_form" / "predictions.jsonl"
    if rk.exists():
        _, top1 = predict(pool, test, 1)
        llm = {r["id"]: r for r in read_jsonl(rk)}
        by_id = dict(zip([r["id"] for r in test], top1))
        agree = sum(1 for i, lbl in by_id.items()
                    if i in llm and llm[i]["pred"] == lbl)
        n = sum(1 for i in by_id if i in llm)
        print(f"\n  retrieval_k10 predictions equal to the top-1 "
              f"neighbour's label: {agree / n * 100:.1f}% ({agree:,d}/{n:,d})")
        print("    High agreement means the language model is mostly")
        print("    repeating the retriever rather than reasoning over it.")


if __name__ == "__main__":
    main()
