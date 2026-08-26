"""
Stage 3a - prompt templates.

Every prompt in this project lives here and carries a version string that
is recorded in each run's metadata. A prompt edited between two runs is a
changed experiment; if the version does not move, the comparison is void.

The templates differ in exactly one axis - what context accompanies the
query - so that a difference between baselines is attributable:

  zero_shot   task instruction + the 77 label names
  few_shot    the same, plus k fixed exemplars, identical for every item
  retrieval   the same, plus the k nearest exemplars to this query

`python src/prompts.py` renders each one and reports token budgets. Run it
before spending GPU time; a prompt that does not fit is cheaper to find on
a laptop.
"""

import json
import re
from pathlib import Path

SPLITS = Path("eval/splits")

VERSIONS = {
    "system": "sys-v1",
    "zero_shot": "zs-v1",
    "few_shot": "fs-v1",
    "retrieval": "rag-v1",
    "bare": "bare-v1",
}

SYSTEM = (
    "You are a banking customer-service intent classifier. "
    "Reply with exactly one intent label from the list and nothing else. "
    "Do not explain. Do not invent labels."
)

# For the fine-tuned model. No label list: after training the label space
# lives in the weights, so listing 77 labels every call would be paying
# ~390 tokens per prediction for information the model already has. 58
# tokens against zero-shot's 447.
#
# It also means the fine-tuned model receives strictly LESS information in
# its prompt than every prompting baseline it is compared against. That
# handicap runs in the conservative direction - if it wins anyway the
# result is clean, and if it loses it lost while disadvantaged. Same
# principle as choosing lenient parsing for the headline in stage 2.
SYSTEM_BARE = (
    "You are a banking customer-service intent classifier. "
    "Reply with exactly one intent label and nothing else."
)


def load_labels():
    p = SPLITS / "train_pool.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run src/01b_build_dataset.py")
    with open(p, encoding="utf-8") as f:
        return sorted({json.loads(line)["label"] for line in f})


def load_pool():
    with open(SPLITS / "train_pool.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ------------------------------------------------------------ label block
def render_labels(labels, order="alpha", seed=None):
    """The label inventory as it appears in the prompt.

    order='alpha'   fixed alphabetical - reproducible, what a practitioner
                    would write, and what every run uses by default
    order='shuffle' a fixed permutation from `seed` - used only to measure
                    how much accuracy moves with label order, which is a
                    prompting noise source with no training analogue
    """
    labels = list(labels)
    if order == "shuffle":
        if seed is None:
            raise ValueError("order='shuffle' requires a seed")
        import random
        random.Random(seed).shuffle(labels)
    elif order != "alpha":
        raise ValueError(f"unknown order {order!r}")
    return "\n".join(labels)


def _exemplar_block(examples):
    return "\n\n".join(f"Query: {e['text']}\nIntent: {e['label']}"
                       for e in examples)


# --------------------------------------------------------------- templates
def zero_shot(query, labels, order="alpha", seed=None):
    user = (f"Intent labels:\n{render_labels(labels, order, seed)}\n\n"
            f"Query: {query}\nIntent:")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


def few_shot(query, labels, examples, order="alpha", seed=None):
    user = (f"Intent labels:\n{render_labels(labels, order, seed)}\n\n"
            f"Examples:\n{_exemplar_block(examples)}\n\n"
            f"Query: {query}\nIntent:")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


def retrieval(query, labels, retrieved, order="alpha", seed=None):
    """Identical body to few_shot; the exemplars are chosen per query.

    Kept as a separate function so the version string differs and the two
    baselines can never be confused in a run manifest, even though the
    rendered text has the same shape.
    """
    user = (f"Intent labels:\n{render_labels(labels, order, seed)}\n\n"
            f"Similar past queries:\n{_exemplar_block(retrieved)}\n\n"
            f"Query: {query}\nIntent:")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


# --------------------------------------------------------- exemplar choice
def fixed_exemplars(pool, k, seed=20260821):
    # seed selects WHICH examples are shown, not how many. Varying it with
    # k held constant separates "few-shot hurts" from "this particular
    # draw of examples hurts" - one draw cannot tell them apart.
    """k exemplars, identical for every test item, spread across classes.

    Drawn from train_pool only - dev and test are never shown to a model in
    a prompt. Round-robin over classes so a small k does not concentrate on
    whichever classes happen to be frequent.
    """
    import random
    rng = random.Random(seed)
    by_label = {}
    for row in pool:
        by_label.setdefault(row["label"], []).append(row)
    labels = sorted(by_label)
    rng.shuffle(labels)
    for lbl in labels:
        rng.shuffle(by_label[lbl])

    picked, i = [], 0
    while len(picked) < k:
        lbl = labels[i % len(labels)]
        depth = i // len(labels)
        if depth < len(by_label[lbl]):
            picked.append(by_label[lbl][depth])
        i += 1
        if i > k * len(labels) + len(labels):
            break
    return picked[:k]


# ------------------------------------------------------------- token count
def make_counter():
    """Real tokenizer if available, crude fallback otherwise (flagged).

    The template is exercised once here rather than trusted. Loading a
    tokenizer succeeds even when apply_chat_template will later fail for a
    missing optional dependency (jinja2), so a construction-time try/except
    around the load alone catches nothing.
    """
    def estimate(messages):
        chars = sum(len(m["content"]) for m in messages)
        return int(chars / 3.6)  # rough English/code average

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

        def count(messages):
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            return len(tok(text).input_ids)

        count([{"role": "system", "content": "x"},
               {"role": "user", "content": "y"}])  # prove the path works
        return count, "Qwen/Qwen3-1.7B tokenizer"
    except Exception as e:  # noqa: BLE001
        return estimate, f"ESTIMATE ONLY ({type(e).__name__}: {e})"


def main():
    labels = load_labels()
    pool = load_pool()
    query = "I am still waiting on my card?"
    count, source = make_counter()

    print(f"labels {len(labels)}   pool {len(pool):,d}   token source: {source}")

    print(f"\n{'=' * 68}\nrendered: zero_shot ({VERSIONS['zero_shot']})\n{'=' * 68}")
    zs = zero_shot(query, labels)
    body = zs[1]["content"]
    head, tail = body.split("\n", 4)[:2], body.rsplit("\n", 4)[-4:]
    print("  [system] " + zs[0]["content"][:70] + "...")
    print("  [user]   " + "\n           ".join(head))
    print("           ... 73 more labels ...")
    print("           " + "\n           ".join(tail))

    print(f"\n{'=' * 68}\ntoken budget\n{'=' * 68}")
    print(f"  {'template':16s} {'k':>3s}  {'tokens':>7s}  {'vs zero-shot':>12s}")
    zs_n = count(zs)
    print(f"  {'zero_shot':16s} {'-':>3s}  {zs_n:7d}  {'-':>12s}")
    for k in (5, 10, 20, 77):
        ex = fixed_exemplars(pool, k)
        n = count(few_shot(query, labels, ex))
        print(f"  {'few_shot':16s} {k:3d}  {n:7d}  {n / zs_n:11.2f}x")
    print(f"\n  bare query, no label list: "
          f"{count([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'Query: {query}'}])} tokens")

    print(f"\n{'=' * 68}\nfixed exemplar spread ({VERSIONS['few_shot']})\n{'=' * 68}")
    for k in (5, 10, 20, 77):
        ex = fixed_exemplars(pool, k)
        n_lab = len({e["label"] for e in ex})
        print(f"  k={k:3d}  distinct classes covered: {n_lab:3d} / 77"
              f"   ({n_lab / 77 * 100:4.1f}%)")
    print("\n  Coverage, not context length, is what constrains few-shot here.")
    print("  A manageable prompt (k=20) shows the model a quarter of the label")
    print("  space; full coverage needs k=77, one example each, at ~4.9x the")
    print("  tokens of zero-shot and still only one example per class.")

    print(f"\n{'=' * 68}\nlabel order sensitivity check\n{'=' * 68}")
    alpha = render_labels(labels)
    for seed in (1, 2, 3):
        shuf = render_labels(labels, order="shuffle", seed=seed)
        same_first = shuf.split("\n")[0]
        print(f"  seed {seed}: first label {same_first!r}   "
              f"identical to alpha: {shuf == alpha}")
    print("\n  Order is fixed alphabetical for every reported run. The shuffle")
    print("  path exists to measure order as a noise source, not to tune.")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------ fine-tuned
def bare(query):
    """Prompt for the fine-tuned model: no label list, no exemplars."""
    return [{"role": "system", "content": SYSTEM_BARE},
            {"role": "user", "content": f"Query: {query}\nIntent:"}]
