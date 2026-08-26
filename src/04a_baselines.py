"""
Stage 3 - prompting baselines.

Five configurations, each run under both decoding regimes fixed in stage 2.
Nothing here is tuned: every prompt comes from src/prompts.py at a recorded
version, and no configuration is adjusted after seeing its score. The
majority-class floor exists so every later number has something trivial to
beat.

Resumability matters. A Kaggle session dies at ~12 hours and the k=77
few-shot pass is measured in hours, so predictions are appended to disk as
they are produced and a rerun skips whatever is already there.

What is stored per item, and why:

  raw          the model's exact output. Every scoring convention in
               stage 2 - lenient, strict, conditional - is recomputable
               from it without spending another GPU-second.
  top2_margin  constrained only. Stage 3 measured batching noise at
               0.04-0.17 logits; items with a margin below that are
               decided by fp16 reduction order rather than by the model.
               Storing the margin makes that population identifiable
               after the fact instead of hypothetical.
  top5         enough of the score vector to diagnose a failure without
               committing 237,000 floats per configuration to git.

    python src/04a_baselines.py --configs zero_shot --limit 64
    python src/04a_baselines.py                      # everything
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must be set before torch initialises CUDA. Long-prompt configs allocate
# and free multi-gigabyte caches per item; without expandable segments the
# allocator fragments and OOMs with gigabytes nominally free.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts as P  # noqa: E402
from evaluator import evaluate_constrained, evaluate_free_form, score  # noqa: E402
from runner import write_env  # noqa: E402

SPLITS = Path("eval/splits")
# Overridden by --runs-dir. On Kaggle the repo clone is destroyed with the
# session, so results must be written to /kaggle/working to survive - a
# multi-hour pass that vanishes at session end is worse than not running it.
RUNS = Path("reports/runs")
BATCH_SIZE = 32          # pinned: see docs/03_inference.md, batching noise
RETRIEVAL_K = 10
CHECKPOINT_EVERY = 256

CONFIGS = {
    "majority":      {"kind": "majority"},
    "zero_shot":     {"kind": "prompt", "template": "zero_shot"},
    # The untuned model on the fine-tuned model's prompt. Without this,
    # "LoRA took the model from 47.1% to 93.6%" mixes two changes: the
    # adapter, and dropping the 77-label list from the prompt. This holds
    # the prompt constant so the adapter is the only difference.
    "bare":          {"kind": "prompt", "template": "bare"},
    "few_shot_k5":   {"kind": "prompt", "template": "few_shot", "k": 5},
    "few_shot_k20":  {"kind": "prompt", "template": "few_shot", "k": 20},
    # Same k, different draws. Separates "exemplars hurt" from "this draw
    # of exemplars hurts", which one seed cannot distinguish.
    "few_shot_k20_s2": {"kind": "prompt", "template": "few_shot", "k": 20,
                        "seed": 20260822},
    "few_shot_k20_s3": {"kind": "prompt", "template": "few_shot", "k": 20,
                        "seed": 20260823},
    "few_shot_k77":  {"kind": "prompt", "template": "few_shot", "k": 77},
    "retrieval_k10": {"kind": "prompt", "template": "retrieval",
                      "k": RETRIEVAL_K},
}


def rule(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}", flush=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_dir(config, regime, limit=None):
    """Limited runs get their own directory.

    A --limit run draws a shuffled subsample; a full run walks test order.
    Sharing a directory would let the full run resume on top of rows that
    correspond to different items - no error, silently mismatched output.
    """
    suffix = f"__limit{limit}" if limit else ""
    return RUNS / f"{config}__{regime}{suffix}"


def set_runs_dir(path):
    global RUNS
    RUNS = Path(path)
    RUNS.mkdir(parents=True, exist_ok=True)


def already_done(path):
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


# ---------------------------------------------------------------- retrieval
def build_retriever(pool):
    """Nearest pool examples for a query, by the same features stage 1 used.

    Note this baseline shares the fine-tuned model's memorisation channel:
    for the 13.8% of test items with a near-duplicate twin in the pool,
    retrieval surfaces that twin with its label attached. That is the point
    - it makes retrieval the fairest comparator, since both methods use the
    pool and both benefit. Clean-subset reporting applies to it identically.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X = vec.fit_transform([r["text"] for r in pool])
    nn = NearestNeighbors(n_neighbors=RETRIEVAL_K, metric="cosine").fit(X)

    def retrieve(queries):
        Q = vec.transform(queries)
        dist, idx = nn.kneighbors(Q)
        out = []
        for row_d, row_i in zip(dist, idx):
            # deterministic order: similarity desc, then id, so ties do not
            # depend on sklearn's internal ordering (see stage 1)
            pairs = sorted(zip(1 - row_d, row_i),
                           key=lambda p: (-p[0], pool[p[1]]["id"]))
            out.append([pool[i] for _, i in pairs])
        return out

    return retrieve


# ----------------------------------------------------------------- prompts
def build_prompts(cfg, rows, labels, pool, tok, render):
    """cfg may carry a 'seed' overriding which exemplars are drawn."""
    tmpl = cfg["template"]
    if tmpl == "bare":
        msgs = [P.bare(r["text"]) for r in rows]
    elif tmpl == "zero_shot":
        msgs = [P.zero_shot(r["text"], labels) for r in rows]
    elif tmpl == "few_shot":
        ex = P.fixed_exemplars(pool, cfg["k"], seed=cfg.get("seed", 20260821))
        msgs = [P.few_shot(r["text"], labels, ex) for r in rows]
    elif tmpl == "retrieval":
        retrieve = build_retriever(pool)
        neighbours = retrieve([r["text"] for r in rows])
        msgs = [P.retrieval(r["text"], labels, nb)
                for r, nb in zip(rows, neighbours)]
    else:
        raise ValueError(f"unknown template {tmpl}")
    return [render(tok, m) for m in msgs]


# --------------------------------------------------------------- baselines
def run_majority(rows, pool, labels):
    from collections import Counter
    top = Counter(r["label"] for r in pool).most_common(1)[0][0]
    return [evaluate_constrained(top, r["label"], labels) | {"id": r["id"],
            "query": r["text"]} for r in rows]


def run_free_form(model, tok, texts, rows, labels, out_path, batch_size):
    """batch_size is the ceiling; the caller lowers it for long prompts."""
    from inference import batched_generate
    done = already_done(out_path)
    if done >= len(rows):
        print(f"  free-form already complete ({done} rows)")
        return read_jsonl(out_path)
    print(f"  free-form: {len(rows) - done} remaining of {len(rows)}")

    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for start in range(done, len(rows), CHECKPOINT_EVERY):
            block = slice(start, min(start + CHECKPOINT_EVERY, len(rows)))
            raws = batched_generate(model, tok, texts[block],
                                    batch_size=batch_size, progress=False)
            for r, raw in zip(rows[block], raws):
                rec = evaluate_free_form(raw, r["label"], labels,
                                         exclude=r["text"])
                rec |= {"id": r["id"], "query": r["text"]}
                f.write(json.dumps(rec, ensure_ascii=False,
                                   sort_keys=True) + "\n")
            f.flush()
            print(f"    {min(start + CHECKPOINT_EVERY, len(rows))}/{len(rows)}",
                  flush=True)
    return read_jsonl(out_path)


def run_constrained(model, tok, texts, rows, labels, out_path, chunk=None):
    from inference import encode_labels, score_labels_cached
    done = already_done(out_path)
    if done >= len(rows):
        print(f"  constrained already complete ({done} rows)")
        return read_jsonl(out_path)
    print(f"  constrained: {len(rows) - done} remaining of {len(rows)}")

    label_ids = encode_labels(tok, labels)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for i in range(done, len(rows)):
            scores = score_labels_cached(model, tok, texts[i], labels,
                                         label_ids, chunk_size=chunk)
            order = sorted(range(len(labels)), key=lambda j: -scores[j])
            pred = labels[order[0]]
            rec = evaluate_constrained(pred, rows[i]["label"], labels)
            rec |= {
                "id": rows[i]["id"],
                "query": rows[i]["text"],
                "top2_margin": round(scores[order[0]] - scores[order[1]], 4),
                "top5": [[labels[j], round(scores[j], 4)] for j in order[:5]],
            }
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            if (i + 1) % CHECKPOINT_EVERY == 0:
                f.flush()
                # Long-prompt configs churn multi-GB caches; release
                # cached blocks periodically so fragmentation does not
                # accumulate over thousands of items.
                import torch
                torch.cuda.empty_cache()
                rate = (i + 1 - done) / (time.time() - t0)
                eta = (len(rows) - i - 1) / max(rate, 1e-9) / 60
                print(f"    {i + 1}/{len(rows)}  {rate:.1f} items/s  "
                      f"eta {eta:.0f} min", flush=True)
    return read_jsonl(out_path)


# -------------------------------------------------------------------- main
def summarise(name, regime, records, clean_ids):
    s = score(records)
    c = score(records, subset_ids=clean_ids)
    print(f"  {name:16s} {regime:12s} "
          f"acc {s['accuracy'] * 100:5.1f}%   "
          f"clean {c['accuracy'] * 100:5.1f}%   "
          f"wrong {s['wrong_label_rate'] * 100:5.1f}%   "
          f"unparseable {s['unparseable_rate'] * 100:5.1f}%")
    return {"full": s, "clean": c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--regimes", nargs="*", default=["free_form", "constrained"])
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test on a shuffled subsample")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--runs-dir", default=str(RUNS),
                    help="on Kaggle use /kaggle/working/runs so results "
                         "survive the session")
    args = ap.parse_args()
    set_runs_dir(args.runs_dir)
    print(f"runs dir: {RUNS.resolve()}")

    labels = P.load_labels()
    pool = P.load_pool()
    test = read_jsonl(SPLITS / "test.jsonl")
    near = {r["id"]: r for r in read_jsonl(SPLITS / "near_dup_test.jsonl")}
    manifest = json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))
    thr = manifest["near_dup_threshold_recorded"]
    clean_ids = [i for i, r in near.items() if r["twin_sim"] < thr]

    if args.limit:
        # test.jsonl is class-ordered; a prefix is one intent
        import random
        test = list(test)
        random.Random(20260821).shuffle(test)
        test = test[:args.limit]

    print(f"labels {len(labels)}  test {len(test):,d}  "
          f"clean subset {len(clean_ids):,d}  batch {args.batch_size}")

    needs_model = any(CONFIGS[c]["kind"] == "prompt" for c in args.configs)
    model = tok = render = None
    if needs_model:
        from inference import load_model, render as _render
        model, tok = load_model(args.model)
        render = _render
        print(f"loaded {args.model}")

    results = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        rule(name)

        if cfg["kind"] == "majority":
            d = run_dir(name, "constrained", args.limit)
            d.mkdir(parents=True, exist_ok=True)
            recs = run_majority(test, pool, labels)
            with open(d / "predictions.jsonl", "w", encoding="utf-8",
                      newline="\n") as f:
                for r in recs:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            results[(name, "constrained")] = summarise(
                name, "constrained", recs, clean_ids)
            continue

        texts = build_prompts(cfg, test, labels, pool, tok, render)
        n_tok = max(len(tok(t).input_ids) for t in texts[:64])
        from inference import plan_generation_batch, plan_label_chunk
        gen_batch = min(args.batch_size, plan_generation_batch(n_tok))
        chunk = plan_label_chunk(n_tok, len(labels))
        print(f"  prompt tokens: {n_tok}   generation batch {gen_batch}   "
              f"label chunk {chunk}")

        for regime in args.regimes:
            d = run_dir(name, regime, args.limit)
            d.mkdir(parents=True, exist_ok=True)
            write_env(d)
            (d / "meta.json").write_text(json.dumps({
                "config": name, "regime": regime, "model": args.model,
                "batch_size": gen_batch, "label_chunk": chunk,
                "prompt_tokens": n_tok, "n_items": len(test),
                "limit": args.limit,
                "prompt_versions": P.VERSIONS,
                "splits_manifest_sha_core": manifest["sha256_core"],
            }, indent=2), encoding="utf-8", newline="\n")

            out = d / "predictions.jsonl"
            if regime == "free_form":
                recs = run_free_form(model, tok, texts, test, labels, out,
                                     gen_batch)
            else:
                recs = run_constrained(model, tok, texts, test, labels, out,
                                       chunk)
            results[(name, regime)] = summarise(name, regime, recs, clean_ids)

    rule("summary")
    print(f"  {'config':16s} {'regime':12s} "
          f"{'acc':>7s} {'clean':>7s} {'wrong':>7s} {'unparse':>8s}")
    for (name, regime), r in results.items():
        s, c = r["full"], r["clean"]
        print(f"  {name:16s} {regime:12s} "
              f"{s['accuracy'] * 100:6.1f}% {c['accuracy'] * 100:6.1f}% "
              f"{s['wrong_label_rate'] * 100:6.1f}% "
              f"{s['unparseable_rate'] * 100:7.1f}%")


if __name__ == "__main__":
    main()
