#!/usr/bin/env python3
"""
DATAFORM Training Artifact Generator

Generates ONNX training artifacts for DATAFORM's on-device LoRA training.
This script runs ONCE offline to produce the files needed by OrtTrainingManager.

Supports any HuggingFace causal LM model. Auto-detects chat template and
LoRA target modules from the model architecture.

Output directory structure:
    training_artifacts/
        training_model.onnx     - Forward + backward pass model
        eval_model.onnx         - Evaluation model
        optimizer_model.onnx    - AdamW optimizer state
        checkpoint/             - Initial checkpoint state
        vocab.json              - BPE vocabulary
        merges.txt              - BPE merge rules
        model_config.json       - Model config for runtime

Usage:
    pip install -r requirements.txt
    python generate_artifacts.py --output <profile_path>/training_artifacts
    python generate_artifacts.py --output ./training_artifacts --model Qwen/Qwen2.5-1.5B
    python generate_artifacts.py --output ./training_artifacts --model mistralai/Mistral-7B-v0.1
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ONNX training artifacts for DATAFORM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-1.5B)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for training artifacts",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=None,
        help="Override LoRA target modules (default: auto-detect)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        choices=["chatml", "llama", "mistral", "zephyr", "raw"],
        help="Override chat template (default: auto-detect)",
    )
    return parser.parse_args()


# --- Chat template detection ---

CHAT_TEMPLATE_PATTERNS = {
    "chatml": ["<|im_start|>", "im_start"],
    "llama": ["[INST]", "[/INST]"],
    "mistral": ["[INST]", "[/INST]"],
    "zephyr": ["<|user|>", "<|assistant|>"],
}

CHAT_TEMPLATE_CONFIGS = {
    "chatml": {
        "system_prefix": "<|im_start|>system\n",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
        "assistant_suffix": "<|im_end|>\n",
        "special_tokens": ["<|im_start|>", "<|im_end|>"],
    },
    "llama": {
        "system_prefix": "<<SYS>>\n",
        "system_suffix": "\n<</SYS>>\n\n",
        "user_prefix": "[INST] ",
        "user_suffix": " [/INST]\n",
        "assistant_prefix": "",
        "assistant_suffix": "\n",
        "special_tokens": [],
    },
    "mistral": {
        "system_prefix": "[INST] ",
        "system_suffix": "\n",
        "user_prefix": "[INST] ",
        "user_suffix": " [/INST]\n",
        "assistant_prefix": "",
        "assistant_suffix": "</s>\n",
        "special_tokens": [],
    },
    "zephyr": {
        "system_prefix": "<|system|>\n",
        "system_suffix": "</s>\n",
        "user_prefix": "<|user|>\n",
        "user_suffix": "</s>\n",
        "assistant_prefix": "<|assistant|>\n",
        "assistant_suffix": "</s>\n",
        "special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
    },
    "raw": {
        "system_prefix": "System: ",
        "system_suffix": "\n\n",
        "user_prefix": "User: ",
        "user_suffix": "\n",
        "assistant_prefix": "Assistant: ",
        "assistant_suffix": "\n",
        "special_tokens": [],
    },
}


def detect_chat_template(tokenizer) -> str:
    """Auto-detect chat template from tokenizer configuration."""
    # Check tokenizer.chat_template string if available
    chat_template_str = getattr(tokenizer, "chat_template", None) or ""

    for template_name, patterns in CHAT_TEMPLATE_PATTERNS.items():
        for pattern in patterns:
            if pattern in chat_template_str:
                print(f"  Detected chat template: {template_name} (from tokenizer.chat_template)")
                return template_name

    # Check if special tokens exist in vocab
    vocab = tokenizer.get_vocab()
    if "<|im_start|>" in vocab and "<|im_end|>" in vocab:
        print(f"  Detected chat template: chatml (from vocab special tokens)")
        return "chatml"

    if "<|user|>" in vocab and "<|assistant|>" in vocab:
        print(f"  Detected chat template: zephyr (from vocab special tokens)")
        return "zephyr"

    # Check model name heuristics
    model_name = getattr(tokenizer, "name_or_path", "").lower()
    if "qwen" in model_name:
        print(f"  Detected chat template: chatml (Qwen family default)")
        return "chatml"
    if "llama" in model_name:
        print(f"  Detected chat template: llama (Llama family default)")
        return "llama"
    if "mistral" in model_name or "mixtral" in model_name:
        print(f"  Detected chat template: mistral (Mistral family default)")
        return "mistral"
    if "zephyr" in model_name:
        print(f"  Detected chat template: zephyr (Zephyr family default)")
        return "zephyr"
    if "phi" in model_name:
        print(f"  Detected chat template: chatml (Phi family default)")
        return "chatml"

    print(f"  Could not auto-detect chat template, defaulting to chatml")
    return "chatml"


# --- LoRA target module detection ---

# Known attention projection layer names across architectures
KNOWN_LORA_TARGETS = [
    # Standard HuggingFace naming (Llama, Mistral, Qwen, Phi, etc.)
    ("q_proj", "v_proj"),
    # GPT-2/GPT-J style
    ("q_attn", "v_attn"),
    # BLOOM style
    ("query_key_value",),
    # GPT-NeoX style
    ("query", "value"),
    # Falcon style
    ("query_key_value",),
]


def detect_lora_targets(model) -> list:
    """Auto-detect LoRA target modules by scanning model architecture."""
    named_modules = dict(model.named_modules())
    module_names = set()

    for name, module in named_modules.items():
        # Only look at Linear layers
        if isinstance(module, torch.nn.Linear):
            # Get the short name (last component)
            short_name = name.split(".")[-1]
            module_names.add(short_name)

    # Try each known pattern
    for target_set in KNOWN_LORA_TARGETS:
        if all(t in module_names for t in target_set):
            targets = list(target_set)
            print(f"  Detected LoRA targets: {targets}")
            return targets

    # Fallback: look for any module with 'q' and 'v' or 'query' and 'value' in name
    q_candidates = [n for n in module_names if n.startswith("q_") or n == "query"]
    v_candidates = [n for n in module_names if n.startswith("v_") or n == "value"]
    if q_candidates and v_candidates:
        targets = [q_candidates[0], v_candidates[0]]
        print(f"  Detected LoRA targets (heuristic): {targets}")
        return targets

    # Last resort: use q_proj + v_proj and hope for the best
    print(f"  WARNING: Could not auto-detect LoRA targets, defaulting to ['q_proj', 'v_proj']")
    return ["q_proj", "v_proj"]


def load_model_with_lora(model_name: str, lora_rank: int, lora_alpha: int,
                          target_modules: list = None):
    """Load base model and apply LoRA adapters."""
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # ORT training needs float32
        trust_remote_code=True,
    )

    # Auto-detect targets if not specified
    if target_modules is None:
        print("Auto-detecting LoRA target modules...")
        target_modules = detect_lora_targets(model)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,  # No dropout for inference stability
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha}) to {target_modules}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, target_modules


class _OnnxExportWrapper(torch.nn.Module):
    """Wrapper that maps positional args to keyword args for ONNX export.

    Many HuggingFace models (Qwen2, Llama, etc.) require keyword arguments
    in their forward() method, but torch.onnx.export passes positional args.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.logits


def export_to_onnx(model, output_dir: str, max_seq_length: int):
    """Export model to ONNX format for ORT training."""
    import onnx as onnx_module

    onnx_model_path = os.path.join(output_dir, "model.onnx")

    print(f"Exporting to ONNX: {onnx_model_path}")

    # Use training mode so batch norm / dropout are included in graph
    model.train()
    wrapper = _OnnxExportWrapper(model)
    wrapper.train()

    # Create dummy inputs matching the training batch shape
    batch_size = 1
    seq_length = max_seq_length
    dummy_input_ids = torch.ones(batch_size, seq_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    dummy_labels = torch.ones(batch_size, seq_length, dtype=torch.long)

    # Export with training-compatible settings
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask, dummy_labels),
        onnx_model_path,
        input_names=["input_ids", "attention_mask", "labels"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "labels": {0: "batch_size", 1: "seq_length"},
            "logits": {0: "batch_size", 1: "seq_length"},
        },
        opset_version=17,
        do_constant_folding=False,  # Keep for training
        training=torch.onnx.TrainingMode.TRAINING,
        keep_initializers_as_inputs=False,
    )

    size_mb = os.path.getsize(onnx_model_path) / 1e6
    print(f"ONNX model exported: {size_mb:.1f} MB")

    # Validate the model has initializers (parameters)
    onnx_model = onnx_module.load(onnx_model_path)
    num_initializers = len(onnx_model.graph.initializer)
    print(f"  Graph initializers: {num_initializers}")
    if num_initializers == 0:
        print("  WARNING: No initializers in ONNX model — parameters may not have been exported!")

    return onnx_model_path


def generate_training_artifacts(onnx_model_path: str, output_dir: str, model):
    """Generate ORT training artifacts from the ONNX model."""
    import onnx as onnx_module
    import onnx.checker
    from onnxruntime.training import artifacts

    print("Generating ORT training artifacts...")

    # Load ONNX model structure only (no external weight data) to avoid >2GB
    # protobuf limit. We use nominal_checkpoint=True so actual weights aren't
    # needed in the proto — generate_artifacts just needs the graph topology.
    onnx_model = onnx_module.load(onnx_model_path, load_external_data=False)

    # Get the ONNX graph parameter names for matching
    onnx_param_names = {init.name for init in onnx_model.graph.initializer}
    print(f"  ONNX initializers: {len(onnx_param_names)}")

    # Identify trainable parameters (LoRA weights only)
    # Match PyTorch param names to ONNX initializer names
    requires_grad = []
    frozen = []
    for name, param in model.named_parameters():
        # ONNX export may mangle names; try the PyTorch name as-is first
        if name in onnx_param_names:
            if param.requires_grad:
                requires_grad.append(name)
            else:
                frozen.append(name)

    # If no matches, try matching without the module prefix (wrapper adds "model.")
    if not requires_grad and not frozen:
        print("  No direct name matches -- trying ONNX initializer names directly...")
        for onnx_name in onnx_param_names:
            # Check if any PyTorch trainable param name is a suffix of the ONNX name
            is_trainable = False
            for pt_name, param in model.named_parameters():
                if onnx_name.endswith(pt_name) or pt_name.endswith(onnx_name):
                    if param.requires_grad:
                        is_trainable = True
                    break
            if is_trainable:
                requires_grad.append(onnx_name)
            else:
                frozen.append(onnx_name)

    print(f"  Trainable parameters: {len(requires_grad)}")
    print(f"  Frozen parameters: {len(frozen)}")

    # Monkey-patch onnx.checker.check_model to handle >2GB models
    # ORT training v1.15 calls check_model internally, which fails on large protos
    _original_check_model = onnx.checker.check_model
    def _patched_check_model(model_or_path, full_check=False):
        if isinstance(model_or_path, str):
            return _original_check_model(model_or_path, full_check)
        try:
            return _original_check_model(model_or_path, full_check)
        except ValueError as e:
            if "too large" in str(e) or "2GiB" in str(e):
                print("  (Skipping ONNX model validation for >2GB model)")
                return
            raise
    onnx.checker.check_model = _patched_check_model

    try:
        # Generate artifacts
        artifacts.generate_artifacts(
            onnx_model,
            optimizer=artifacts.OptimType.AdamW,
            loss=artifacts.LossType.CrossEntropyLoss,
            artifact_directory=output_dir,
            requires_grad=requires_grad,
            frozen_params=frozen,
            nominal_checkpoint=True,
        )
    finally:
        # Restore original check_model
        onnx.checker.check_model = _original_check_model

    # Verify outputs
    expected_files = [
        "training_model.onnx",
        "eval_model.onnx",
        "optimizer_model.onnx",
        "checkpoint",
    ]
    for f in expected_files:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            if os.path.isdir(path):
                print(f"  {f}/ (directory)")
            else:
                size_mb = os.path.getsize(path) / 1e6
                print(f"  {f} ({size_mb:.1f} MB)")
        else:
            print(f"  WARNING: {f} not found!")

    # Clean up the intermediate ONNX model
    if os.path.exists(onnx_model_path):
        os.remove(onnx_model_path)
        print(f"  Cleaned up intermediate {os.path.basename(onnx_model_path)}")


def copy_tokenizer_files(model_name: str, output_dir: str):
    """Copy tokenizer vocab.json and merges.txt to the artifacts directory."""
    print("Copying tokenizer files...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Save tokenizer files
    tokenizer_dir = os.path.join(output_dir, "_tokenizer_tmp")
    tokenizer.save_pretrained(tokenizer_dir)

    # Copy vocab.json and merges.txt
    for filename in ["vocab.json", "merges.txt"]:
        src = os.path.join(tokenizer_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size_kb = os.path.getsize(dst) / 1024
            print(f"  {filename} ({size_kb:.0f} KB)")
        else:
            print(f"  WARNING: {filename} not found in tokenizer output")

    # Cleanup temp directory
    shutil.rmtree(tokenizer_dir, ignore_errors=True)

    return tokenizer


def write_model_config(tokenizer, model_name: str, output_dir: str,
                        chat_template: str, target_modules: list,
                        lora_rank: int, lora_alpha: int, max_seq_length: int):
    """Write model configuration JSON for DATAFORM runtime."""
    print("Writing model configuration...")

    # Resolve special token IDs safely
    vocab = tokenizer.get_vocab()

    def get_token_id(token_str, fallback=-1):
        if token_str in vocab:
            return vocab[token_str]
        return fallback

    special_tokens = {
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }

    # Add template-specific special tokens
    template_config = CHAT_TEMPLATE_CONFIGS.get(chat_template, {})
    for token_str in template_config.get("special_tokens", []):
        key = token_str.strip("<|>")
        tid = get_token_id(token_str)
        if tid >= 0:
            special_tokens[key] = tid

    # Always try common special tokens
    for token_str, key in [("<|im_start|>", "im_start"), ("<|im_end|>", "im_end"),
                            ("<|endoftext|>", "endoftext")]:
        tid = get_token_id(token_str)
        if tid >= 0:
            special_tokens[key] = tid

    config = {
        "model_name": model_name,
        "chat_template": chat_template,
        "chat_template_config": template_config,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_length": max_seq_length,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "special_tokens": special_tokens,
    }

    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  model_config.json written")
    print(f"    Chat template: {chat_template}")
    print(f"    LoRA targets:  {target_modules}")
    print(f"    Vocab size:    {tokenizer.vocab_size}")
    print(f"    Special tokens: {special_tokens}")


def main():
    args = parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATAFORM Training Artifact Generator")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"LoRA rank:      {args.lora_rank}")
    print(f"LoRA alpha:     {args.lora_alpha}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"Output:         {output_dir}")
    print("=" * 60)

    # Step 1: Auto-detect chat template
    print("\nStep 1: Detecting chat template...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = args.chat_template or detect_chat_template(tokenizer)

    # Step 2: Load model with LoRA (auto-detects targets if not specified)
    print("\nStep 2: Loading model with LoRA...")
    model, target_modules = load_model_with_lora(
        args.model, args.lora_rank, args.lora_alpha, args.target_modules
    )

    # Step 3: Export to ONNX
    print("\nStep 3: Exporting to ONNX...")
    onnx_path = export_to_onnx(model, str(output_dir), args.max_seq_length)

    # Step 4: Generate ORT training artifacts
    print("\nStep 4: Generating ORT training artifacts...")
    generate_training_artifacts(onnx_path, str(output_dir), model)

    # Step 5: Copy tokenizer files
    print("\nStep 5: Copying tokenizer files...")
    copy_tokenizer_files(args.model, str(output_dir))

    # Step 6: Write model config (replaces old tokenizer_config.json)
    print("\nStep 6: Writing model configuration...")
    write_model_config(tokenizer, args.model, str(output_dir),
                        chat_template, target_modules,
                        args.lora_rank, args.lora_alpha, args.max_seq_length)

    print()
    print("=" * 60)
    print("Artifact generation complete!")
    print(f"Output directory: {output_dir}")
    print()
    print("To use with DATAFORM:")
    print(f"  1. Copy {output_dir} to your DATAFORM profile path")
    print(f"  2. Launch DATAFORM - it will detect the artifacts automatically")
    print(f"  3. Select 'Local' provider in Settings to use the trained model")
    print("=" * 60)


if __name__ == "__main__":
    main()
