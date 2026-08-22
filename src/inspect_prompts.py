"""
Look at the prompt before believing the number.

few_shot_k77 scored 5.7% - forty points below using no examples at all, and
barely above the 1.3% majority floor. Constrained decoding gives the same
5.7%, so it is not an output-format artefact: the model genuinely ranks the
wrong labels highest.

That is an extraordinary claim. Adding correctly-labelled examples should
not destroy a classifier. Two explanations fit equally well:

  1. long irrelevant exemplar blocks genuinely poison this model
  2. our prompt is malformed and the model is answering a different
     question than we think

We have counted these prompts' tokens and measured their accuracy, but
nobody has read one. This prints them in full, alongside the model's actual
output, so explanation 2 can be ruled out by inspection before anything is
written up.

It also tests one specific mechanism: whether predictions track the
*position* of exemplars - copying the last one seen (recency), the first
(primacy), or neither.

    python src/inspect_prompts.py --config few_shot_k77 --n 3
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts as P  # noqa: E402

RUNS = Path("reports/runs")
SPLITS = Path("eval/splits")
KS = {"few_shot_k5": 5, "few_shot_k20": 20, "few_shot_k77": 77}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def show_prompt(text, head=18, tail=14):
    lines = text.split("\n")
    if len(lines) <= head + tail:
        print("    " + "\n    ".join(lines))
        return
    print("    " + "\n    ".join(lines[:head]))
    print(f"    ... [{len(lines) - head - tail} lines omitted] ...")
    print("    " + "\n    ".join(lines[-tail:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="few_shot_k77")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    k = KS[args.config]
    labels = P.load_labels()
    pool = P.load_pool()
    test = read_jsonl(SPLITS / "test.jsonl")
    ex = P.fixed_exemplars(pool, k)
    ex_labels = [e["label"] for e in ex]

    pred_path = RUNS / f"{args.config}__free_form" / "predictions.jsonl"
    preds = {r["id"]: r for r in read_jsonl(pred_path)} \
        if pred_path.exists() else {}

    print(f"{args.config}: k={k}, {len(set(ex_labels))} distinct exemplar "
          f"labels, order is round-robin over shuffled classes")

    print(f"\n{'=' * 68}\nfull prompt, as sent to the model\n{'=' * 68}")
    for r in test[:1]:
        msgs = P.few_shot(r["text"], labels, ex)
        print(f"  [system]\n    {msgs[0]['content']}")
        print(f"  [user]")
        show_prompt(msgs[1]["content"])

    print(f"\n{'=' * 68}\nqueries and what came back\n{'=' * 68}")
    for r in test[:args.n]:
        rec = preds.get(r["id"])
        print(f"\n  query : {r['text']}")
        print(f"  gold  : {r['label']}")
        if rec:
            print(f"  raw   : {str(rec.get('raw')).strip()[:100]!r}")
            print(f"  pred  : {rec.get('pred')}   [{rec.get('match_mode')}]")
            if rec.get("pred") in ex_labels:
                pos = ex_labels.index(rec["pred"])
                print(f"          ^ was exemplar #{pos + 1} of {k}")

    if not preds:
        print("\n  (no predictions file - prompt inspection only)")
        return

    print(f"\n{'=' * 68}\ndoes the prediction track exemplar position?\n{'=' * 68}")
    rows = [r for r in preds.values() if r.get("pred") in ex_labels]
    pos = Counter(ex_labels.index(r["pred"]) for r in rows)
    n_in = len(rows)
    print(f"  predictions matching some exemplar label: {n_in:,d} / "
          f"{len(preds):,d}")
    if n_in:
        first = sum(v for p, v in pos.items() if p < k * 0.1)
        last = sum(v for p, v in pos.items() if p >= k * 0.9)
        print(f"    in first 10% of exemplar list: {first / n_in * 100:5.1f}%"
              f"   (uniform would be 10%)")
        print(f"    in last  10% of exemplar list: {last / n_in * 100:5.1f}%"
              f"   (uniform would be 10%)")
        print("\n    most-predicted exemplars, by position:")
        for p, v in pos.most_common(6):
            print(f"      #{p + 1:3d}/{k}  {ex_labels[p]:42s} {v:5,d}")

    print("\n  Strong primacy or recency means the model is copying by")
    print("  position rather than classifying. A flat spread means it is")
    print("  classifying badly, which is a different finding.")


if __name__ == "__main__":
    main()
