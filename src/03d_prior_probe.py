"""
Is the model answering the query, or reciting a prior?

few_shot accuracy collapses as exemplars are added, and the collapse target
is a refund intent in all three k=20 draws - `Refund_not_showing_up` in one,
`request_refund` in the other two, including a draw where `request_refund`
was never shown as an exemplar. Recency was refuted (1 of 3). The surviving
hypothesis: the model holds a strong prior over a few intents, and a long
exemplar block drowns out the query, leaving that prior exposed.

Direct test. Replace the customer's message with something carrying no
information and score all 77 labels:

  - if the top label under a contentless query matches the label the
    config actually collapsed onto, the model was reciting its prior
  - if a contentless query gives something else, the collapse depends on
    the query after all and this explanation fails

Also measured: query sensitivity. Score labels for N real queries and count
how many distinct argmax labels come back. Zero-shot should span many; a
collapsed config should span very few. That number is the collapse,
measured directly on the model's beliefs rather than inferred from
accuracy.

    !python src/03d_prior_probe.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts as P  # noqa: E402
from inference import encode_labels, load_model, render, score_labels_cached  # noqa: E402

SPLITS = Path("eval/splits")
N_QUERIES = 40
SAMPLE_SEED = 20260821

# Contentless stand-ins for a customer message.
BLANKS = ["", "hello", "hi there", "...", "I have a question"]

CONFIGS = {
    "zero_shot": {"k": None},
    "few_shot_k20": {"k": 20, "seed": 20260821},
    "few_shot_k20_s2": {"k": 20, "seed": 20260822},
    "few_shot_k20_s3": {"k": 20, "seed": 20260823},
    "few_shot_k77": {"k": 77, "seed": 20260821},
}

# What each config actually collapsed onto, from the completed runs.
OBSERVED = {
    "zero_shot": None,
    "few_shot_k20": "Refund_not_showing_up",
    "few_shot_k20_s2": "request_refund",
    "few_shot_k20_s3": "request_refund",
    "few_shot_k77": "pending_transfer",
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build(cfg, query, labels, pool):
    if cfg["k"] is None:
        return P.zero_shot(query, labels)
    ex = P.fixed_exemplars(pool, cfg["k"], seed=cfg["seed"])
    return P.few_shot(query, labels, ex)


def main():
    labels, pool = P.load_labels(), P.load_pool()
    test = read_jsonl(SPLITS / "test.jsonl")

    import random
    queries = list(test)
    random.Random(SAMPLE_SEED).shuffle(queries)
    queries = queries[:N_QUERIES]   # spread across classes, not one intent

    model, tok = load_model()
    label_ids = encode_labels(tok, labels)

    def top1(messages):
        text = render(tok, messages)
        scores = score_labels_cached(model, tok, text, labels, label_ids)
        return labels[max(range(len(labels)), key=lambda i: scores[i])]

    print(f"{'config':18s} {'contentless-query top label':34s} "
          f"{'distinct':>9s} {'top share':>10s}  matches observed?")

    for name, cfg in CONFIGS.items():
        blanks = Counter(top1(build(cfg, b, labels, pool)) for b in BLANKS)
        prior_label, _ = blanks.most_common(1)[0]

        real = Counter(top1(build(cfg, q["text"], labels, pool))
                       for q in queries)
        _, n_top = real.most_common(1)[0]

        obs = OBSERVED.get(name)
        verdict = "-" if obs is None else ("YES" if prior_label == obs else "no")
        print(f"{name:18s} {prior_label:34s} {len(real):6d}/{N_QUERIES} "
              f"{n_top / N_QUERIES * 100:9.1f}%  {verdict}")

    print(f"\n  'distinct' = distinct argmax labels over {N_QUERIES} real")
    print("  queries spanning many classes. A model reading the query")
    print("  produces many; a collapsed one produces very few.")
    print("\n  'matches observed' compares the contentless-query prediction")
    print("  to the label that config actually collapsed onto across all")
    print("  3,080 test items. YES means the collapse is the prior.")


if __name__ == "__main__":
    main()
