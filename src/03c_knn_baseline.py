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

This costs seconds on a laptop. If it lands near 82.7%, the honest reading
of retrieval_k10 changes completely.

    python src/03c_knn_baseline.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from evaluator import evaluate_constrained, score  # noqa: E402

SPLITS = Path("eval/splits")
RUNS = Path("reports/runs")
NEIGHBOURS = [1, 3, 5, 10]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_ids():
    m = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    thr = m["near_dup_threshold_recorded"]
    return [r["id"] for r in read_jsonl(SPLITS / "near_dup_test.jsonl")
            if r["twin_sim"] < thr]


def predict(pool, test, k):
    """k-NN over char n-gram TF-IDF. Ties broken by nearer neighbour, then
    by pool id, so the result does not depend on sklearn's ordering."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X = vec.fit_transform([r["text"] for r in pool])
    Q = vec.transform([r["text"] for r in test])
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
        best = max(sorted(weight), key=lambda lbl: weight[lbl])
        preds.append(best)
    return preds, top1_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    pool = read_jsonl(SPLITS / "train_pool.jsonl")
    test = read_jsonl(SPLITS / "test.jsonl")
    labels = sorted({r["label"] for r in pool})
    clean = clean_ids()
    print(f"pool {len(pool):,d}   test {len(test):,d}   "
          f"clean {len(clean):,d}   labels {len(labels)}")

    print(f"\n  {'config':12s} {'acc':>8s} {'clean':>8s}   "
          f"(no model, no GPU)")
    results = {}
    for k in NEIGHBOURS:
        preds, top1 = predict(pool, test, k)
        recs = [evaluate_constrained(p, r["label"], labels)
                | {"id": r["id"], "query": r["text"]}
                for p, r in zip(preds, test)]
        d = runs / f"knn_k{k}__constrained"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "predictions.jsonl", "w", encoding="utf-8",
                  newline="\n") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False,
                                   sort_keys=True) + "\n")
        (d / "meta.json").write_text(json.dumps({
            "config": f"knn_k{k}", "regime": "constrained",
            "model": "tfidf char_wb 3-5 + kNN", "n_items": len(test),
        }, indent=2), encoding="utf-8", newline="\n")

        s, c = score(recs), score(recs, subset_ids=clean)
        results[k] = (s, c)
        print(f"  knn_k{k:<7d} {s['accuracy'] * 100:7.1f}% "
              f"{c['accuracy'] * 100:7.1f}%")

    # How much of retrieval_k10's answer is just the top neighbour's label?
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
