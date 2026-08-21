"""
Stage 3b - inference.

Two paths, because stage 2 fixed two decoding regimes:

  batched_generate        free-form; the model writes, we parse
  score_labels_*          constrained; score all 77 labels, take the argmax

Both are optimisations of something obvious and slow, and both fail
silently when wrong. Left-padding a decoder incorrectly produces fluent
garbage, not an error. A mishandled KV cache produces plausible
log-probabilities, not an error. So each fast path ships with the naive
implementation beside it and a check that they agree.

Why the constrained path needs care. Naively, scoring 77 labels for 3,080
test items is 237,160 forward passes over ~460-token sequences. Sharing the
prompt's KV cache across the 77 label continuations cuts the work per item
from ~35,000 token-positions to ~1,400 - roughly 26x - which is the
difference between hours and minutes per configuration, and therefore the
difference between the ablation grid fitting in Kaggle's weekly quota and
not.

The cache-expansion API has moved across transformers versions, so three
strategies are tried in order and the working one is reported. If none
work, the naive path is used and says so. Correctness never depends on
which strategy succeeded - that is what verify_scoring_equivalence is for.
"""

import time

import torch

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"


# ------------------------------------------------------------------ setup
def load_model(model_id=DEFAULT_MODEL, adapter_path=None, dtype=torch.float16):
    """Load base model, optionally with a LoRA adapter attached.

    fp16 not bf16: the Kaggle T4 is sm75 and has no bf16 tensor cores.
    See docs/00_environment.md - torch.cuda.is_bf16_supported() lies here.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="cuda")

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # fold LoRA in; faster inference

    model.eval()
    return model, tok


def render(tok, messages):
    """Chat template with thinking disabled.

    Qwen3 can emit a reasoning block, which our free-form parser would
    correctly refuse as ambiguous - scoring the model near zero for
    reasons unrelated to classification.
    """
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


# ------------------------------------------------------- free-form path
@torch.no_grad()
def batched_generate(model, tok, prompts, batch_size=32, max_new_tokens=16,
                     progress=True):
    """Greedy generation over a list of rendered prompt strings.

    Left padding is mandatory: with right padding, a decoder generates from
    a position preceded by pad tokens and the output is fluent nonsense
    with no error raised. Verified against the unbatched path by
    verify_generation_equivalence.
    """
    original_side = tok.padding_side
    tok.padding_side = "left"
    outputs = []
    try:
        t0 = time.time()
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=False).to(model.device)
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id)
            new = gen[:, enc.input_ids.shape[1]:]
            outputs.extend(tok.batch_decode(new, skip_special_tokens=True))
            if progress and (i // batch_size) % 10 == 0:
                done = min(i + batch_size, len(prompts))
                rate = done / max(time.time() - t0, 1e-9)
                print(f"    generate {done:5d}/{len(prompts)}  "
                      f"{rate:.1f} items/s", flush=True)
    finally:
        tok.padding_side = original_side
    return outputs


# ------------------------------------------------------ constrained path
def encode_labels(tok, labels):
    """Token ids for each label continuation, no special tokens."""
    ids = [tok(lbl, add_special_tokens=False).input_ids for lbl in labels]
    if any(len(x) == 0 for x in ids):
        raise ValueError("a label tokenised to nothing")
    return ids


@torch.no_grad()
def _token_logprobs(logits, targets):
    """Log-probs of `targets` given `logits`, without a full-vocab copy.

    log_softmax(x).gather(t) allocates another [B, T, V] tensor. At vocab
    151,936 that is gigabytes for a handful of numbers. gather - logsumexp
    computes the same thing with [B, T] intermediates.
    """
    lse = torch.logsumexp(logits.float(), dim=-1)
    picked = logits.gather(2, targets.unsqueeze(-1)).squeeze(-1).float()
    return picked - lse


@torch.no_grad()
def score_labels_naive(model, tok, prompt, labels, label_ids=None,
                       batch_size=8):
    """Reference implementation. Slow, obvious, assumed correct.

    Builds prompt+label for every label and scores the continuation. Used
    to validate the fast path and as a fallback if cache expansion fails.

    Only the final `longest + 1` logit positions are ever needed, so they
    are the only ones computed where transformers supports `logits_to_keep`.
    Computing all of them and slicing afterwards is a multi-gigabyte
    allocation for a dozen useful numbers.
    """
    label_ids = label_ids or encode_labels(tok, labels)
    p_ids = tok(prompt, return_tensors="pt").input_ids[0]
    n_prompt = len(p_ids)
    pad = tok.pad_token_id

    scores = []
    for i in range(0, len(labels), batch_size):
        chunk = label_ids[i:i + batch_size]
        longest = max(len(c) for c in chunk)
        rows, masks, lens = [], [], []
        for c in chunk:
            rows.append(p_ids.tolist() + c + [pad] * (longest - len(c)))
            masks.append([1] * (n_prompt + len(c)) + [0] * (longest - len(c)))
            lens.append(len(c))
        ids = torch.tensor(rows, device=model.device)
        att = torch.tensor(masks, device=model.device)

        keep = longest + 1
        try:
            logits = model(ids, attention_mask=att,
                           logits_to_keep=keep).logits
            kept = logits[:, :-1]                      # positions n_prompt-1 ..
        except TypeError:
            logits = model(ids, attention_mask=att).logits
            kept = logits[:, n_prompt - 1:n_prompt - 1 + longest]

        tgt = ids[:, n_prompt:n_prompt + longest]
        tok_lp = _token_logprobs(kept, tgt)
        pos = torch.arange(longest, device=model.device).unsqueeze(0)
        mask = pos < torch.tensor(lens, device=model.device).unsqueeze(1)
        scores.extend((tok_lp * mask).sum(dim=1).tolist())
        del logits, kept, tok_lp
    return scores


def _expand_cache(past, n):
    """Repeat a batch-1 KV cache to batch n. Returns (cache, strategy_name).

    Raises if no strategy works, so the caller can fall back rather than
    proceeding with a cache of the wrong shape.
    """
    if hasattr(past, "batch_repeat_interleave"):
        try:
            import copy
            c = copy.deepcopy(past)
            c.batch_repeat_interleave(n)
            return c, "batch_repeat_interleave"
        except Exception:  # noqa: BLE001
            pass

    if hasattr(past, "to_legacy_cache") and hasattr(
            type(past), "from_legacy_cache"):
        try:
            legacy = past.to_legacy_cache()
            grown = tuple(tuple(t.expand(n, *t.shape[1:]).contiguous()
                                for t in layer) for layer in legacy)
            return type(past).from_legacy_cache(grown), "legacy_cache"
        except Exception:  # noqa: BLE001
            pass

    if hasattr(past, "layers"):
        try:
            import copy
            c = copy.deepcopy(past)
            for layer in c.layers:
                layer.keys = layer.keys.expand(
                    n, *layer.keys.shape[1:]).contiguous()
                layer.values = layer.values.expand(
                    n, *layer.values.shape[1:]).contiguous()
            return c, "layers"
        except Exception:  # noqa: BLE001
            pass

    raise RuntimeError("no working cache-expansion strategy")


@torch.no_grad()
def score_labels_cached(model, tok, prompt, labels, label_ids=None):
    """Score every label against one shared prompt KV cache.

    The prompt is encoded once. The first label token's log-prob comes from
    the prompt's final logits; the rest come from a single forward over the
    label tokens with the expanded cache.
    """
    label_ids = label_ids or encode_labels(tok, labels)
    n = len(label_ids)
    dev = model.device
    pad = tok.pad_token_id

    p_ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    n_prompt = p_ids.shape[1]
    out = model(p_ids, use_cache=True)
    last = out.logits[:, -1, :].float()
    lp_first = (last - torch.logsumexp(last, dim=-1, keepdim=True))[0]

    longest = max(len(c) for c in label_ids)
    padded = torch.full((n, longest), pad, dtype=torch.long, device=dev)
    lengths = torch.tensor([len(c) for c in label_ids], device=dev)
    for r, c in enumerate(label_ids):
        padded[r, :len(c)] = torch.tensor(c, device=dev)

    totals = lp_first[padded[:, 0]].clone()

    if longest > 1:
        cache, _ = _expand_cache(out.past_key_values, n)
        feed = padded[:, :-1]
        att = torch.ones((n, n_prompt + feed.shape[1]), dtype=torch.long,
                         device=dev)
        logits = model(feed, attention_mask=att,
                       past_key_values=cache).logits
        tgt = padded[:, 1:]
        step_lp = _token_logprobs(logits, tgt)
        pos = torch.arange(longest - 1, device=dev).unsqueeze(0)
        mask = pos < (lengths - 1).unsqueeze(1)
        totals = totals + (step_lp * mask).sum(dim=1)

    return totals.tolist()


def cache_strategy(model, tok, prompt, labels):
    """Which expansion strategy this environment supports, or None."""
    ids = encode_labels(tok, labels)
    p = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(p, use_cache=True)
    try:
        _, name = _expand_cache(out.past_key_values, len(ids))
        return name
    except RuntimeError:
        return None


# ------------------------------------------------------------ validation
def verify_generation_equivalence(model, tok, prompts, max_new_tokens=16,
                                  batch_size=8):
    """Batched output must equal unbatched output, item for item.

    A left-padding mistake shows up here and nowhere else.
    """
    batched = batched_generate(model, tok, prompts, batch_size=batch_size,
                               max_new_tokens=max_new_tokens, progress=False)
    single = batched_generate(model, tok, prompts, batch_size=1,
                              max_new_tokens=max_new_tokens, progress=False)
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(batched, single))
             if a.strip() != b.strip()]
    return {"n": len(prompts), "mismatches": len(diffs),
            "examples": diffs[:3]}


def verify_scoring_equivalence(model, tok, prompt, labels, tol=0.05):
    """Cached scores must match naive scores, and rank identically.

    Tolerance is on absolute log-prob; fp16 accumulation over a shared
    cache will not reproduce the naive path bit for bit. The argmax
    agreeing matters more than the values, since argmax is what we score.
    """
    ids = encode_labels(tok, labels)
    t0 = time.time()
    naive = score_labels_naive(model, tok, prompt, labels, ids)
    t_naive = time.time() - t0
    t0 = time.time()
    cached = score_labels_cached(model, tok, prompt, labels, ids)
    t_cached = time.time() - t0

    deltas = [abs(a - b) for a, b in zip(naive, cached)]
    return {
        "max_abs_delta": max(deltas),
        "mean_abs_delta": sum(deltas) / len(deltas),
        "within_tolerance": max(deltas) <= tol,
        "argmax_agrees": (naive.index(max(naive))
                          == cached.index(max(cached))),
        "top1_naive": labels[naive.index(max(naive))],
        "top1_cached": labels[cached.index(max(cached))],
        "naive_s": t_naive,
        "cached_s": t_cached,
        "speedup": t_naive / max(t_cached, 1e-9),
    }


# ------------------------------------------------------------ diagnosis
@torch.no_grad()
def diagnose_padding(model, tok, prompts, dtype_note=""):
    """Why do batched and unbatched greedy decodes differ?

    Two candidate causes demand different responses:

      position handling  a real bug; left padding must shift position ids
                         so the first real token sits at position 0
      fp16 accumulation  batching changes matmul shapes and therefore
                         reduction order; a near-tie between two labels
                         can flip. Not fixable, but measurable - and it
                         makes batch size a nuisance parameter that must
                         be pinned and recorded like a seed.

    Reported per prompt: the largest logit disagreement at the first
    generated position, whether the argmax flips, and the top-2 gap. A
    flip on a large gap is a bug. A flip on a gap smaller than the logit
    delta is arithmetic.
    """
    side = tok.padding_side
    tok.padding_side = "left"
    try:
        batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        blogits = model(**batch).logits[:, -1, :].float()
        rows = []
        for i, p in enumerate(prompts):
            single = tok(p, return_tensors="pt").to(model.device)
            slogits = model(**single).logits[:, -1, :].float()[0]
            b = blogits[i]
            top2 = torch.topk(slogits, 2).values
            rows.append({
                "i": i,
                "pad_tokens": int((batch.attention_mask[i] == 0).sum()),
                "max_logit_delta": float((b - slogits).abs().max()),
                "argmax_batched": int(b.argmax()),
                "argmax_single": int(slogits.argmax()),
                "flipped": int(b.argmax()) != int(slogits.argmax()),
                "top2_gap": float(top2[0] - top2[1]),
            })
        return rows
    finally:
        tok.padding_side = side
