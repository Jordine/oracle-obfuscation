"""
Per-token activation oracle probing for Gemma-2-9B-IT.

Uses the exact same oracle query pattern as ncao_ref/utils.py:
  - Raw prompt: "Layer: {N}\n ? \n{question}" (NO chat template)
  - Injection via apply_oracle_math: h + ||h|| * (v / ||v||)
  - Injection at layer 1, special token " ?"

Target model prompt DOES use chat template (that's what the model sees).

Usage:
    python scripts/oracle_probe_tokens.py \
        --prompt "Mom is sleeping in the next room..." \
        --question "What language or cultural context is being expressed here?"
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
from contextlib import contextmanager

import torch
from torch import nn, Tensor
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Config
# ============================================================

MODEL_NAME = "google/gemma-2-9b-it"
ORACLE_LORA = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"
NUM_LAYERS = 42
INJECTION_LAYER = 1
SOURCE_LAYER = 21  # 50% of 42 layers — default for AO


# ============================================================
# From ncao_ref/utils.py — exact same logic
# ============================================================

def apply_oracle_math(h: Tensor, v: Tensor) -> Tensor:
    """Norm-matched addition: h' = h + ||h|| * (v / ||v||)"""
    v_unit = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    return h + h.norm(dim=-1, keepdim=True) * v_unit


def get_layer(model, layer_idx):
    """Get transformer layer module, handling PeftModel wrapping."""
    for attr in [
        'base_model.model.model.layers',
        'model.model.layers',
        'model.layers',
    ]:
        parts = attr.split('.')
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            return obj[layer_idx]
        except (AttributeError, IndexError):
            continue
    raise ValueError(f"Could not find layer {layer_idx}")


class EarlyStopException(Exception):
    pass


def collect_activations(model, inputs, layer_idx):
    """Extract activations at a given layer. Returns [1, seq_len, d_model]."""
    layer = get_layer(model, layer_idx)
    result = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        result["act"] = h.detach()
        raise EarlyStopException()

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    except EarlyStopException:
        pass
    finally:
        handle.remove()

    return result["act"]


def query_activation_oracle(
    model,
    tokenizer,
    vector: Tensor,
    question: str,
    source_layer: int = SOURCE_LAYER,
    injection_layer: int = INJECTION_LAYER,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> str:
    """Query the oracle about a single activation vector.

    Follows the exact pattern from ncao_ref/utils.py:
      - Raw prompt (no chat template): "Layer: {N}\n ? \n{question}"
      - Inject at " ?" token position using apply_oracle_math
    """
    prompt = f"Layer: {source_layer}\n ? \n{question}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Find " ?" token position
    special_ids = tokenizer.encode(" ?", add_special_tokens=False)
    if len(special_ids) != 1:
        raise ValueError(f"' ?' encodes to {len(special_ids)} tokens, expected 1")
    special_id = special_ids[0]

    positions = (inputs.input_ids[0] == special_id).nonzero(as_tuple=True)[0].tolist()
    if not positions:
        # Fallback: try just "?"
        special_id = tokenizer.encode("?", add_special_tokens=False)[-1]
        positions = (inputs.input_ids[0] == special_id).nonzero(as_tuple=True)[0].tolist()
    if not positions:
        raise ValueError("Could not find injection position in oracle prompt")

    # Build injection hook
    inj_layer = get_layer(model, injection_layer)
    vec = vector.unsqueeze(0) if vector.dim() == 1 else vector  # [1, d_model] or [n, d_model]

    def injection_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1:
            return out  # Skip KV-cache steps (single token generation)
        h_new = h.clone()
        for i, pos in enumerate(positions[:vec.shape[0]]):
            if pos < h_new.shape[1]:
                h_new[:, pos:pos+1, :] = apply_oracle_math(
                    h[:, pos:pos+1, :],
                    vec[i:i+1].to(h.device, h.dtype)
                )
        if isinstance(out, tuple):
            return (h_new,) + out[1:]
        return h_new

    handle = inj_layer.register_forward_hook(injection_hook)
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Per-token activation oracle probing")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--question", type=str,
                        default="Describe what the model is representing at this point.")
    parser.add_argument("--source_layer", type=int, default=SOURCE_LAYER,
                        help=f"Layer to extract activations from (default {SOURCE_LAYER})")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--token_start", type=int, default=0)
    parser.add_argument("--token_end", type=int, default=None)
    parser.add_argument("--segment", action="store_true",
                        help="Also query oracle on all selected tokens at once")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model: {MODEL_NAME}")
    print(f"Oracle: {ORACLE_LORA}")
    print(f"Source layer: {args.source_layer}")
    print(f"Injection layer: {INJECTION_LAYER}")
    print(f"Question: {args.question}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )

    # Load oracle LoRA
    print("Loading oracle LoRA...")
    model = PeftModel.from_pretrained(model, ORACLE_LORA, adapter_name="oracle", is_trainable=False)
    model.set_adapter("oracle")

    # Format TARGET prompt with chat template
    messages = [{"role": "user", "content": args.prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    print(f"Target prompt:\n{formatted}\n")

    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in input_ids]

    print(f"Tokens ({len(tokens)}):")
    for i, tok in enumerate(tokens):
        print(f"  [{i:3d}] {repr(tok)}")
    print()

    # Collect activations from BASE model (no oracle LoRA)
    print("Collecting activations (base model)...")
    with model.disable_adapter():
        all_acts = collect_activations(model, inputs, args.source_layer)
    model.set_adapter("oracle")
    print(f"Shape: {all_acts.shape}\n")

    # Token range
    start = args.token_start
    if start < 0:
        start = len(tokens) + start
    end = args.token_end if args.token_end is not None else len(tokens)

    # Per-token oracle queries
    print("=" * 70)
    print("PER-TOKEN ORACLE RESPONSES")
    print("=" * 70)

    results = []
    for i in range(start, end):
        vec = all_acts[0, i, :]  # [d_model]
        response = query_activation_oracle(
            model, tokenizer, vec, args.question,
            source_layer=args.source_layer,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        print(f"[{i:3d}] {repr(tokens[i]):30s} → {response}")
        results.append({"idx": i, "token": tokens[i], "response": response})

    # Optional segment query
    segment_response = None
    if args.segment and end > start:
        print()
        print("=" * 70)
        print(f"SEGMENT RESPONSE (tokens {start}:{end})")
        print("=" * 70)

        # For segment: inject multiple activations at multiple ? positions
        # Build prompt with multiple ? tokens
        num_pos = end - start
        segment_prompt = f"Layer: {args.source_layer}\n" + " ?" * num_pos + " \n" + args.question
        seg_inputs = tokenizer(segment_prompt, return_tensors="pt").to(device)

        special_ids = tokenizer.encode(" ?", add_special_tokens=False)
        special_id = special_ids[0] if len(special_ids) == 1 else tokenizer.encode("?", add_special_tokens=False)[-1]
        positions = (seg_inputs.input_ids[0] == special_id).nonzero(as_tuple=True)[0].tolist()

        segment_vecs = all_acts[0, start:end, :]  # [num_pos, d_model]
        inj_layer = get_layer(model, INJECTION_LAYER)

        def seg_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] <= 1:
                return out
            h_new = h.clone()
            for j, pos in enumerate(positions[:segment_vecs.shape[0]]):
                if pos < h_new.shape[1]:
                    h_new[:, pos:pos+1, :] = apply_oracle_math(
                        h[:, pos:pos+1, :],
                        segment_vecs[j:j+1].to(h.device, h.dtype)
                    )
            return (h_new,) + out[1:] if isinstance(out, tuple) else h_new

        handle = inj_layer.register_forward_hook(seg_hook)
        try:
            with torch.no_grad():
                output = model.generate(
                    **seg_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        finally:
            handle.remove()

        segment_response = tokenizer.decode(
            output[0][seg_inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        print(f"→ {segment_response}")

    # Save
    if args.output:
        out = {
            "prompt": args.prompt,
            "formatted_prompt": formatted,
            "question": args.question,
            "source_layer": args.source_layer,
            "tokens": tokens,
            "per_token": results,
        }
        if segment_response:
            out["segment_response"] = segment_response
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
