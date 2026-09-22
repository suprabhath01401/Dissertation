"""Runs ON the Warwick GPU cluster — reads input JSON, generates with the given
--model, writes output JSON. Model-agnostic: called by both mixtral_inference.sbatch
and llama_inference.sbatch with a different --model each time.

Called by e.g.:
  python backend/routing/cluster_inference.py \
    --input-file $INFER_INPUT --output-file $INFER_OUTPUT --model <hf-repo-id>

Uses HuggingFace transformers (available on the cluster via conda/modules).
Falls back to Ollama if OLLAMA_BASE_URL is set in the environment.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Point HuggingFace to the large-disk allocation before any transformers import.
# The sbatch already exports HF_HOME; this is a safety net for direct invocations.
_HF_STORAGE = "/dcs/large/u5754610"
os.environ.setdefault("HF_HOME", _HF_STORAGE)


def _generate_hqq(prompt: str, model_id: str) -> str:
    """Load a locally-cached HQQ-quantized checkpoint and generate a completion."""
    from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("HF_HOME")
    print(f"[cluster_inference] Resolving local path for {model_id}…")
    local_path = snapshot_download(model_id, cache_dir=cache_dir, local_files_only=True)
    print(f"[cluster_inference] Loading from {local_path} via HQQ…")
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = HQQModelForCausalLM.from_quantized(local_path)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


_MIXTRAL_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"


def _generate_mixtral_nf4(prompt: str, model_id: str) -> str:
    """Load Mixtral-8x7B-Instruct-v0.1 via HuggingFace transformers with a
    4-bit NF4 (bitsandbytes) quantization config and generate a completion.

    4-bit NF4 with double quantization keeps the 46.7B-parameter dense
    footprint of the 8x7B MoE checkpoint within a single A40 (48GB VRAM);
    only 2 of 8 experts activate per token, so peak activation memory is
    well below what a dense model of the same parameter count would need.
    BF16 compute (rather than FP16) avoids overflow during quantized
    matmuls on the A40.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[cluster_inference] Loading {model_id} via transformers (4-bit NF4)…")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _generate_transformers(prompt: str, model_id: str) -> str:
    """Load a full-precision/fp16 checkpoint via plain HuggingFace transformers and generate a completion."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[cluster_inference] Loading {model_id} via transformers…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _generate_ollama(prompt: str, model: str, base_url: str) -> str:
    """Generate a completion via an already-running Ollama server on the cluster node."""
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=model, base_url=base_url)
    return str(llm.invoke(prompt))


def main() -> None:
    """CLI entrypoint: read the input JSON, dispatch to the right backend, write the output JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model", default=_MIXTRAL_MODEL_ID)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    query = payload.get("query", "")
    system_prompt = payload.get("system_prompt", "")
    full_prompt = f"{system_prompt}\n\nUser: {query}\n\nAssistant:" if system_prompt else query

    # Multiple generation backends because cluster nodes are provisioned
    # differently per job: prefer an already-running Ollama server when
    # configured (cheapest, no model load); the canonical Mixtral-8x7B
    # checkpoint loads via plain HF transformers with 4-bit NF4
    # quantization (BitsAndBytesConfig); other checkpoints shipped as
    # pre-quantized HQQ weights use the HQQ backend; anything else falls
    # back to plain transformers at fp16.
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "")
    if ollama_url:
        print(f"[cluster_inference] Using Ollama at {ollama_url} (model={args.model})")
        response = _generate_ollama(full_prompt, args.model, ollama_url)
        model_used = f"{args.model} (ollama)"
    elif args.model == _MIXTRAL_MODEL_ID:
        response = _generate_mixtral_nf4(full_prompt, args.model)
        model_used = args.model
    elif "HQQ" in args.model:
        response = _generate_hqq(full_prompt, args.model)
        model_used = args.model
    else:
        response = _generate_transformers(full_prompt, args.model)
        model_used = args.model

    output = {"response": response, "model": model_used}
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_file).write_text(json.dumps(output), encoding="utf-8")
    print(f"[cluster_inference] Written to {args.output_file}")


if __name__ == "__main__":
    main()
