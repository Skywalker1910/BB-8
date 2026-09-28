"""Inspectable, stateless next-token inference for the local BB8 Lab.

This deliberately uses a named sampler rather than claiming bit-for-bit parity
with historical TextGenerator or Transformers.generate implementations.
"""

import math

import torch

from inference.conversation import build_chat_prompt


ENGINE_VERSION = "bb8-lab-v1"


def settings(values):
    if not isinstance(values, dict):
        raise ValueError("settings must be an object")
    result = {
        "strategy": values.get("strategy", "top_p"),
        "temperature": float(values.get("temperature", 0.8)),
        "top_k": int(values.get("top_k", 40)),
        "top_p": float(values.get("top_p", 0.9)),
        "repetition_penalty": float(values.get("repetition_penalty", 1.0)),
        "seed": int(values.get("seed", 42)),
    }
    if result["strategy"] not in {"greedy", "temperature", "top_k", "top_p"}:
        raise ValueError("Unknown decoding strategy")
    for key, low, high in [("temperature", 0.05, 5), ("top_p", 0.001, 1),
                            ("repetition_penalty", 1, 3)]:
        if not math.isfinite(result[key]) or not low <= result[key] <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
    if not 1 <= result["top_k"] <= 200000 or not 0 <= result["seed"] <= 2**32 - 1:
        raise ValueError("Invalid top_k or seed")
    return result


def distributions(logits, seen_ids, options):
    """Return raw, adjusted and selection probabilities from the same logits.

    Order: sign-aware repetition penalty, temperature, then K/P filtering.
    Greedy ignores temperature and yields a one-hot selection distribution.
    """
    scores = logits.detach().to(device="cpu", dtype=torch.float32).flatten().clone()
    if not torch.isfinite(scores).all():
        raise ValueError("Model produced non-finite logits")
    raw = scores.softmax(-1)
    if seen_ids:
        indices = torch.tensor(sorted(set(seen_ids)), dtype=torch.long)
        seen = scores[indices]
        penalty = options["repetition_penalty"]
        scores[indices] = torch.where(seen < 0, seen * penalty, seen / penalty)
    if options["strategy"] != "greedy":
        scores /= options["temperature"]
    adjusted = scores.softmax(-1)
    if options["strategy"] == "greedy":
        chosen = scores.argmax()
        scores.fill_(-torch.inf)
        scores[chosen] = 0
    elif options["strategy"] == "top_k":
        indices = torch.argsort(scores, descending=True, stable=True)
        scores[indices[min(options["top_k"], scores.numel()):]] = -torch.inf
    elif options["strategy"] == "top_p":
        indices = torch.argsort(scores, descending=True, stable=True)
        # Keep the smallest prefix reaching p, including the crossing token.
        cumulative = adjusted[indices].cumsum(0)
        count = min(int(torch.searchsorted(cumulative, options["top_p"]).item()) + 1,
                    scores.numel())
        scores[indices[max(1, count):]] = -torch.inf
    return raw, adjusted, scores.softmax(-1)


def token_info(tokenizer, token_id):
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        piece = tokenizer.convert_ids_to_tokens(token_id)
    else:
        piece = tokenizer.inverse_vocab.get(token_id, "<unknown>")
    return {"id": token_id, "piece": str(piece), "text": tokenizer.decode([token_id])}


def vocab_size(bundle):
    if bundle.backend == "hf_lora":
        return bundle.generator.hf_model.config.vocab_size
    return bundle.generator.model.vocab_size


def prepare(bundle, prompt, mode):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError("Enter a prompt between 1 and 8000 characters")
    tokenizer = bundle.generator.tokenizer
    if mode == "instruction":
        prompt, _ = build_chat_prompt(
            [{"role": "user", "content": prompt}], tokenizer,
            bundle.max_context_tokens, bundle.prompt_style,
        )
    elif mode != "raw":
        raise ValueError("Prompt mode must be raw or instruction")
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > 4096:
        raise ValueError("Prompt must encode to between 1 and 4096 tokens")
    return {"formatted_prompt": prompt, "input_ids": ids,
            "tokens": [token_info(tokenizer, i) for i in ids]}


@torch.inference_mode()
def inspect_step(bundle, ids, options, context_window, step_index=0, forced_token_id=None):
    if not isinstance(ids, list) or not 1 <= len(ids) <= 4160:
        raise ValueError("input_ids must contain between 1 and 4160 tokens")
    size = vocab_size(bundle)
    if any(type(i) is not int or not 0 <= i < size for i in ids):
        raise ValueError("Token IDs do not belong to the selected model vocabulary")
    if not 1 <= context_window <= bundle.max_context_tokens:
        raise ValueError(f"Context window must be 1–{bundle.max_context_tokens}")
    if not 0 <= step_index < 64:
        raise ValueError("A lab run is limited to 64 generated tokens")
    options = settings(options)
    visible = ids[-context_window:]
    inputs = torch.tensor([visible], device=bundle.device)
    generator = bundle.generator
    if bundle.backend == "hf_lora":
        logits = generator.hf_model(input_ids=inputs, use_cache=False).logits[0, -1]
        eos_id = generator.tokenizer.eos_token_id
    else:
        logits = generator.model(inputs)["logits"][0, -1]
        eos_id = generator.tokenizer.vocab.get(generator.tokenizer.eos_token)
    raw, adjusted, selection = distributions(logits, visible, options)
    if forced_token_id is not None:
        if type(forced_token_id) is not int or not 0 <= forced_token_id < size:
            raise ValueError("Invalid manual token ID")
        chosen = forced_token_id
    else:
        rng = torch.Generator(device="cpu").manual_seed(options["seed"] + step_index)
        chosen = int(torch.multinomial(selection, 1, generator=rng).item())
    # Include both raw leaders and selection leaders, plus the actual chosen ID.
    display_ids = set(torch.topk(raw, min(15, size)).indices.tolist())
    selection_ids = torch.topk(selection, min(15, size)).indices.tolist()
    display_ids.update(i for i in selection_ids if selection[i] > 0)
    display_ids.add(chosen)
    rows = []
    for token_id in sorted(display_ids, key=lambda i: (-float(raw[i]), i)):
        rows.append({**token_info(generator.tokenizer, token_id),
                     "logit": float(logits[token_id]),
                     "raw": float(raw[token_id]), "adjusted": float(adjusted[token_id]),
                     "selection": float(selection[token_id]),
                     "eligible": bool(selection[token_id] > 0)})
    def entropy(probabilities):
        positive = probabilities[probabilities > 0]
        return float(-(positive * positive.log2()).sum())
    selected = next(row for row in rows if row["id"] == chosen)
    return {
        "engine": ENGINE_VERSION, "settings": options, "step_index": step_index,
        "context_window": context_window, "context_ids": visible,
        "dropped_tokens": len(ids) - len(visible), "vocab_size": size,
        "candidates": rows, "chosen": selected,
        "selection_method": "manual" if forced_token_id is not None else options["strategy"],
        "eos": chosen == eos_id,
        "eligible_count": int((selection > 0).sum()),
        "entropy": {"raw": entropy(raw), "adjusted": entropy(adjusted),
                    "selection": entropy(selection)},
        "other_mass": {key: max(0.0, 1 - sum(row[key] for row in rows))
                       for key in ("raw", "adjusted", "selection")},
    }
