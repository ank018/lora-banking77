"""
Stage 0c - GPU smoke test. The first thing in this project that touches a GPU.

Two jobs, in order of importance:

  1. Stand up the Kaggle environment and find out what actually breaks there.
     This is the largest schedule risk in the project and it is being run
     before the dataset it will train on is even loaded, deliberately.

  2. Decide the base model from measurement rather than from model cards.
     Qwen3-1.7B is a conventional dense transformer with a well-trodden
     PEFT path. Qwen3.5-2B is newer and stronger but ships a vision encoder
     and a Gated DeltaNet attention variant, which may not expose the
     projection modules a rank ablation depends on.

Nothing here trains anything and nothing here is a result. It reports
facts that determine what stage 3 and stage 4 can assume.

Kaggle setup:
  - Accelerator: GPU T4 x2  (or P100 - the script records which it got)
  - Internet: ON  (required; needs phone verification on the account)

    !git clone -q https://github.com/ank018/lora-banking77.git
    %cd lora-banking77
    !pip install -q peft accelerate
    !python src/00b_smoke_gpu.py
"""

import gc
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CANDIDATES = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3.5-2B"]

# A real Banking77 test item, so the timing is representative.
QUERY = "I am still waiting on my card?"
SYSTEM = ("You are a banking intent classifier. Reply with exactly one "
          "intent label and nothing else.")

VISION_HINTS = ("visual", "vision", "vit", "image_encoder", "patch_embed")
THINK_HINTS = ("<think>", "</think>", "<thinking>")


def rule(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def gb(x):
    return f"{x / 1024 ** 3:.2f} GB"


# ------------------------------------------------------------------ env
def report_environment():
    rule("environment")
    import transformers
    print(f"  torch          {torch.__version__}")
    print(f"  transformers   {transformers.__version__}")
    for name in ("peft", "trl", "bitsandbytes", "accelerate"):
        try:
            print(f"  {name:14s} {__import__(name).__version__}")
        except Exception:  # noqa: BLE001
            print(f"  {name:14s} MISSING")

    if not torch.cuda.is_available():
        print("\n  NO GPU. Set the notebook accelerator and rerun.")
        return None

    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    bf16 = torch.cuda.is_bf16_supported()
    print(f"\n  device         {props.name}")
    print(f"  count          {torch.cuda.device_count()}")
    print(f"  capability     sm{cap[0]}{cap[1]}")
    print(f"  vram           {gb(props.total_memory)}")
    print(f"  cuda           {torch.version.cuda}")
    print(f"  bf16 supported {bf16}")
    if not bf16:
        print("\n  NOTE Turing (sm75) has no bf16 and no flash-attn-2.")
        print("       Every training config in this project must use fp16.")
    return {"device": props.name, "bf16": bf16, "vram": props.total_memory}


# ------------------------------------------------------------- inspect
def linear_modules(model):
    """Suffixes of every nn.Linear, which is what PEFT targets by name."""
    from collections import Counter
    counts = Counter()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            counts[name.split(".")[-1]] += 1
    return counts


def vision_parameter_share(model):
    total = vis = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if any(h in name.lower() for h in VISION_HINTS):
            vis += n
    return total, vis


def build_prompt(tok):
    """Returns (prompt_text, thinking_flag_supported)."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUERY}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        return text, True
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True), False


def score_forced_continuation(model, tok, prompt, label):
    """Can we get a log-prob for a forced continuation?

    Constrained decoding scores all 77 labels this way and takes the argmax.
    If this path does not work, half the evaluation design does not work,
    so it is checked here rather than discovered in stage 3.
    """
    full = prompt + label
    ids_prompt = tok(prompt, return_tensors="pt").input_ids
    ids_full = tok(full, return_tensors="pt").input_ids.to(model.device)
    n_prompt = ids_prompt.shape[1]
    with torch.no_grad():
        logits = model(ids_full).logits.float()
    logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = ids_full[:, 1:]
    tok_lp = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return tok_lp[0, n_prompt - 1:].sum().item()


# --------------------------------------------------------------- probe
def probe(model_id):
    rule(model_id)
    out = {"model": model_id, "loaded": False}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda",
        trust_remote_code=False)
    model.eval()
    out["load_s"] = time.time() - t0
    out["loaded"] = True
    print(f"  loaded in {out['load_s']:.0f}s")
    print(f"  class          {model.__class__.__name__}")

    total, vis = vision_parameter_share(model)
    out["params"] = total
    out["vision_params"] = vis
    print(f"  parameters     {total / 1e9:.2f}B"
          + (f"   ({vis / 1e6:.0f}M in vision modules, "
             f"{vis / total * 100:.1f}%)" if vis else "   (text only)"))
    print(f"  weights vram   {gb(torch.cuda.max_memory_allocated())}")

    counts = linear_modules(model)
    out["linear_modules"] = dict(counts)
    print("\n  nn.Linear suffixes (PEFT target candidates)")
    for name, n in counts.most_common(14):
        print(f"    {name:24s} x{n}")

    standard = {"q_proj", "k_proj", "v_proj", "o_proj"}
    present = standard & set(counts)
    out["standard_attn_projections"] = sorted(present)
    print(f"\n  standard attention projections present: "
          f"{sorted(present) if present else 'NONE'}")
    if present != standard:
        print("    !! rank ablation cannot use the conventional target set")

    prompt, think_flag = build_prompt(tok)
    out["thinking_flag_supported"] = think_flag
    print(f"  enable_thinking accepted by template: {think_flag}")
    print(f"  prompt tokens  {len(tok(prompt).input_ids)}")

    ids = tok(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=24, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    out["gen_s"] = time.time() - t0
    text = tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    out["sample_output"] = text
    out["thinking_leaked"] = any(h in text.lower() for h in THINK_HINTS)
    print(f"\n  generation     {out['gen_s']:.2f}s for 24 tokens "
          f"(single, unbatched)")
    print(f"  output         {text.strip()[:120]!r}")
    if out["thinking_leaked"]:
        print("    !! thinking block leaked into the answer - the free-form "
              "parser would refuse this as ambiguous")

    try:
        lp = score_forced_continuation(model, tok, prompt, "card_arrival")
        out["forced_logprob"] = lp
        print(f"  constrained scoring works: logP(card_arrival) = {lp:.2f}")
    except Exception as e:  # noqa: BLE001
        out["forced_logprob"] = None
        print(f"  !! constrained scoring FAILED: {type(e).__name__}: {e}")

    print(f"  peak vram      {gb(torch.cuda.max_memory_allocated())}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return out


def main():
    env = report_environment()
    if env is None:
        return

    results = []
    for model_id in CANDIDATES:
        try:
            results.append(probe(model_id))
        except Exception as e:  # noqa: BLE001
            print(f"\n  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            results.append({"model": model_id, "loaded": False,
                            "error": f"{type(e).__name__}: {e}"})
            gc.collect()
            torch.cuda.empty_cache()

    rule("summary")
    for r in results:
        if not r.get("loaded"):
            print(f"  {r['model']:22s} FAILED  {r.get('error', '')[:70]}")
            continue
        print(f"  {r['model']:22s} "
              f"{r['params'] / 1e9:.2f}B  "
              f"vision={r['vision_params'] / 1e6:.0f}M  "
              f"attn={'std' if len(r['standard_attn_projections']) == 4 else 'NONSTANDARD'}  "
              f"gen={r['gen_s']:.2f}s  "
              f"think_leak={r['thinking_leaked']}  "
              f"constrained={'ok' if r['forced_logprob'] is not None else 'FAILED'}")

    print("\n  Decision rule stated before the run:")
    print("    take Qwen3.5-2B if it loads on this transformers, exposes the")
    print("    four standard projections, and scores forced continuations;")
    print("    otherwise take Qwen3-1.7B and record the reason in docs.")


if __name__ == "__main__":
    main()
