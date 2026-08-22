"""
Why does few-shot make this worse?

zero_shot free-form scored 45.8%. few_shot_k20 scored 12.0% - three times
worse, from adding twenty labelled examples to a prompt that already listed
all 77 labels.

Hypothesis, written before running this: the exemplars collapse the model's
effective label space onto the labels they demonstrate. If the model only
answers from the k labels it was shown, its accuracy ceiling is k/77
regardless of what the label list says.

The test is direct. Reconstruct the exact exemplars the run used - they are
deterministic given the pool and the seed - and ask what share of
predictions fall inside that set.

  If the hypothesis holds:  predictions concentrate massively on the
                            exemplar labels, and accuracy tracks k/77
  If it fails:              predictions spread over all 77 and the
                            collapse has some other cause

    python src/analyse_fewshot.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts as P  # noqa: E402

RUNS = Path("reports/runs")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def analyse(name, k, labels, pool):
    d = RUNS / f"{name}__free_form"
    path = d / "predictions.jsonl"
    if not path.exists() or sum(1 for _ in open(path, encoding="utf-8")) == 0:
        print(f"\n  {name}: no predictions")
        return

    recs = read_jsonl(path)
    shown = {e["label"] for e in P.fixed_exemplars(pool, k)}
    preds = [r["pred"] for r in recs if r["pred"] is not None]

    inside = sum(1 for p in preds if p in shown)
    acc = sum(1 for r in recs if r["verdict"] == "correct") / len(recs)
    gold_inside = sum(1 for r in recs if r["gold"] in shown) / len(recs)

    print(f"\n  {name}  (k={k}, {len(shown)} distinct exemplar labels)")
    print(f"    accuracy                         {acc * 100:5.1f}%")
    print(f"    ceiling if only exemplar labels  {len(shown) / 77 * 100:5.1f}%")
    print(f"    predictions inside exemplar set  {inside / len(preds) * 100:5.1f}%"
          f"   ({inside:,d} / {len(preds):,d})")
    print(f"    gold inside exemplar set         {gold_inside * 100:5.1f}%"
          f"   <- chance level for the above")
    print(f"    distinct labels predicted        {len(set(preds)):3d} / 77")

    top = Counter(preds).most_common(5)
    print("    most predicted: " + ", ".join(
        f"{lbl}{'*' if lbl in shown else ''} {n}" for lbl, n in top)
        + "      (* = was an exemplar)")

    # Of items whose gold label was demonstrated, how well does it do?
    on_shown = [r for r in recs if r["gold"] in shown]
    off_shown = [r for r in recs if r["gold"] not in shown]
    for tag, group in (("gold WAS shown", on_shown), ("gold NOT shown", off_shown)):
        if group:
            a = sum(1 for r in group if r["verdict"] == "correct") / len(group)
            print(f"    {tag:16s} n={len(group):5,d}  accuracy {a * 100:5.1f}%")


def main():
    labels = P.load_labels()
    pool = P.load_pool()

    print("Do few-shot exemplars collapse the effective label space?")
    print("=" * 68)

    zs = RUNS / "zero_shot__free_form" / "predictions.jsonl"
    if zs.exists():
        recs = read_jsonl(zs)
        preds = [r["pred"] for r in recs if r["pred"] is not None]
        acc = sum(1 for r in recs if r["verdict"] == "correct") / len(recs)
        print(f"\n  zero_shot (control, no exemplars)")
        print(f"    accuracy                   {acc * 100:5.1f}%")
        print(f"    distinct labels predicted  {len(set(preds)):3d} / 77")

    for name, k in (("few_shot_k5", 5), ("few_shot_k20", 20),
                    ("few_shot_k77", 77)):
        analyse(name, k, labels, pool)

    print("\n" + "=" * 68)
    print("  Read 'predictions inside exemplar set' against 'gold inside'.")
    print("  The second is the chance level. A large gap is the collapse.")


if __name__ == "__main__":
    main()
