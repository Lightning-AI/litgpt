# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Pre-flight validation for LitGPT checkpoints.

Usage:
    litgpt validate --checkpoint_dir checkpoints/meta-llama/...
"""

import sys
from pathlib import Path

import torch
import yaml

from litgpt.config import Config
from litgpt.utils import (
    check_valid_checkpoint_dir,
    estimate_model_memory,
    validate_checkpoint,
)


def validate_setup(
    checkpoint_dir: Path,
    model_filename: str = "lit_model.pth",
    dtype: str = "float32",
    training: bool = False,
) -> None:
    """Run pre-flight validation on a checkpoint directory.

    This checks everything without actually running training or generation:
    1. Checkpoint directory structure (required files exist)
    2. Model config loading
    3. Tokenizer loading
    4. Checkpoint key/shape validation against the model
    5. Memory estimation

    Args:
        checkpoint_dir: Path to the checkpoint directory.
        model_filename: Name of the checkpoint file (default: ``lit_model.pth``).
        dtype: Data type for memory estimation (``float32``, ``float16``, ``bfloat16``).
        training: If ``True``, estimate memory for training (includes optimizer states).
    """
    checkpoint_dir = Path(checkpoint_dir)
    print(f"{'=' * 60}")
    print("LitGPT Pre-flight Validation")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"{'=' * 60}\n")

    all_passed = True

    # --- Step 1: Checkpoint directory structure ---
    print("[1/5] Checking checkpoint directory structure...")
    try:
        check_valid_checkpoint_dir(
            checkpoint_dir,
            model_filename=model_filename,
            verbose=True,
            raise_error=True,
        )
        print("  ✓ All required files found.\n")
    except (FileNotFoundError, SystemExit) as e:
        print(f"  ✗ Directory validation failed: {e}\n", file=sys.stderr)
        all_passed = False

    # --- Step 2: Load model config ---
    print("[2/5] Loading model config...")
    config = None
    config_path = checkpoint_dir / "model_config.yaml"
    try:
        if config_path.is_file():
            with open(config_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            config = Config(**config_dict)
            print(f"  ✓ Config loaded: {config.name or 'unnamed'}")
            print(
                f"    n_layer={config.n_layer}, n_embd={config.n_embd}, "
                f"n_head={config.n_head}, vocab_size={config.vocab_size}\n"
            )
        else:
            print(f"  ✗ Config file not found: {config_path}\n", file=sys.stderr)
            all_passed = False
    except Exception as e:
        print(f"  ✗ Failed to load config: {e}\n", file=sys.stderr)
        all_passed = False

    # --- Step 3: Tokenizer ---
    print("[3/5] Checking tokenizer...")
    try:
        from litgpt.tokenizer import Tokenizer

        tokenizer = Tokenizer(checkpoint_dir)
        # Do a simple encode/decode round-trip
        test_text = "Hello"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"  ✓ Tokenizer loaded (backend={tokenizer.backend})")
        print(f'    Round-trip test: "{test_text}" → {tokens.tolist()} → "{decoded}"\n')
    except Exception as e:
        print(f"  ✗ Tokenizer failed: {e}\n", file=sys.stderr)
        all_passed = False

    # --- Step 4: Checkpoint validation ---
    print("[4/5] Validating checkpoint against model...")
    checkpoint_path = checkpoint_dir / model_filename
    if config is not None and checkpoint_path.is_file():
        try:
            from litgpt import GPT

            with torch.device("meta"):
                model = GPT(config)
            result = validate_checkpoint(checkpoint_path, model, verbose=False)
            if result.is_valid:
                print("  ✓ Checkpoint keys and shapes match the model.\n")
            else:
                all_passed = False
                print(f"  ✗ {result.summary()}\n", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ Checkpoint validation error: {e}\n", file=sys.stderr)
            all_passed = False
    elif not checkpoint_path.is_file():
        print(f"  ⊘ Skipped (checkpoint file not found: {checkpoint_path})\n")
    else:
        print("  ⊘ Skipped (config not loaded)\n")

    # --- Step 5: Memory estimation ---
    print("[5/5] Estimating memory requirements...")
    if config is not None:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)
        mem = estimate_model_memory(config, dtype=torch_dtype, training=training)
        print(f"  Estimated parameters: {mem['param_count']:,}")
        print(f"  Parameter memory: {mem['param_memory_gb']:.2f} GB ({dtype})")
        mode_str = "training (params + grads + optimizer)" if training else "inference (params only)"
        print(f"  Estimated total ({mode_str}): {mem['estimated_total_gb']:.2f} GB")
        if mem["available_gpu_memory_gb"] is not None:
            print(f"  Available GPU memory: {mem['available_gpu_memory_gb']:.2f} GB")
            if mem["fits_in_memory"]:
                print("  ✓ Model should fit in GPU memory.\n")
            else:
                print("  ⚠ WARNING: Model may NOT fit in GPU memory!\n", file=sys.stderr)
                all_passed = False
        else:
            print("  ⊘ No GPU detected, skipping memory fit check.\n")
    else:
        print("  ⊘ Skipped (config not loaded)\n")

    # --- Summary ---
    print(f"{'=' * 60}")
    if all_passed:
        print("✓ All validation checks passed!")
    else:
        print("✗ Some validation checks failed. See details above.", file=sys.stderr)
    print(f"{'=' * 60}")

    if not all_passed:
        raise SystemExit(1)
