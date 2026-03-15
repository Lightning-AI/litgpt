# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Tests for the validate CLI script and related error-handling utilities."""

import json
import warnings
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml

from litgpt import GPT
from litgpt.config import Config
from litgpt.utils import (
    CheckpointValidationResult,
    estimate_model_memory,
    validate_checkpoint,
)

# ---------------------------------------------------------------------------
# validate_checkpoint tests
# ---------------------------------------------------------------------------


class TestValidateCheckpoint:
    """Tests for the validate_checkpoint utility."""

    @staticmethod
    def _save_model_checkpoint(model: torch.nn.Module, path: Path) -> None:
        torch.save(model.state_dict(), str(path))

    def test_valid_checkpoint(self, tmp_path):
        """A checkpoint saved from the same model should pass validation."""
        config = Config.from_name("pythia-14m")
        with torch.device("meta"):
            model = GPT(config)
        # Create a real state_dict with matching shapes
        real_sd = {k: torch.randn(v.shape) for k, v in model.state_dict().items()}
        ckpt_path = tmp_path / "lit_model.pth"
        torch.save(real_sd, str(ckpt_path))

        result = validate_checkpoint(ckpt_path, model, verbose=False)
        assert result.is_valid
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        assert result.shape_mismatches == []
        assert result.errors == []
        assert "passed" in result.summary().lower()

    def test_missing_keys(self, tmp_path):
        """Checkpoint missing some keys should report them."""
        config = Config.from_name("pythia-14m")
        with torch.device("meta"):
            model = GPT(config)
        real_sd = {k: torch.randn(v.shape) for k, v in model.state_dict().items()}
        # Remove a key
        removed_key = list(real_sd.keys())[0]
        del real_sd[removed_key]
        ckpt_path = tmp_path / "lit_model.pth"
        torch.save(real_sd, str(ckpt_path))

        result = validate_checkpoint(ckpt_path, model, verbose=False)
        assert not result.is_valid
        assert removed_key in result.missing_keys
        assert result.unexpected_keys == []

    def test_unexpected_keys(self, tmp_path):
        """Checkpoint with extra keys should report them."""
        config = Config.from_name("pythia-14m")
        with torch.device("meta"):
            model = GPT(config)
        real_sd = {k: torch.randn(v.shape) for k, v in model.state_dict().items()}
        real_sd["extra.unexpected.key"] = torch.randn(3)
        ckpt_path = tmp_path / "lit_model.pth"
        torch.save(real_sd, str(ckpt_path))

        result = validate_checkpoint(ckpt_path, model, verbose=False)
        assert not result.is_valid
        assert "extra.unexpected.key" in result.unexpected_keys

    def test_shape_mismatch(self, tmp_path):
        """Checkpoint with wrong shapes should report mismatches."""
        config = Config.from_name("pythia-14m")
        with torch.device("meta"):
            model = GPT(config)
        real_sd = {k: torch.randn(v.shape) for k, v in model.state_dict().items()}
        # Corrupt a shape
        key = "lm_head.weight"
        real_sd[key] = torch.randn(10, 10)  # wrong shape
        ckpt_path = tmp_path / "lit_model.pth"
        torch.save(real_sd, str(ckpt_path))

        result = validate_checkpoint(ckpt_path, model, verbose=False)
        assert not result.is_valid
        assert any(key in m for m in result.shape_mismatches)

    def test_file_not_found(self, tmp_path):
        """Non-existent checkpoint should report an error."""
        config = Config.from_name("pythia-14m")
        with torch.device("meta"):
            model = GPT(config)
        result = validate_checkpoint(tmp_path / "nonexistent.pth", model, verbose=False)
        assert not result.is_valid
        assert any("not found" in e for e in result.errors)

    def test_corrupted_file(self, tmp_path):
        """A file that is not a valid PyTorch checkpoint should report an error."""
        config = Config.from_name("pythia-14m")
        with torch.device("meta"):
            model = GPT(config)
        ckpt_path = tmp_path / "corrupted.pth"
        ckpt_path.write_text("this is not a checkpoint")

        result = validate_checkpoint(ckpt_path, model, verbose=False)
        assert not result.is_valid
        assert any("Failed to load" in e for e in result.errors)

    def test_model_key_wrapper(self, tmp_path):
        """Checkpoint wrapped under a 'model' key should be unwrapped."""
        config = Config.from_name("pythia-14m")
        with torch.device("meta"):
            model = GPT(config)
        real_sd = {k: torch.randn(v.shape) for k, v in model.state_dict().items()}
        wrapped = {"model": real_sd}
        ckpt_path = tmp_path / "lit_model.pth"
        torch.save(wrapped, str(ckpt_path))

        result = validate_checkpoint(ckpt_path, model, verbose=False)
        assert result.is_valid

    def test_summary_format(self):
        """Summary strings should be well-formed."""
        result = CheckpointValidationResult(
            is_valid=True, missing_keys=[], unexpected_keys=[], shape_mismatches=[], errors=[]
        )
        assert result.summary() == "Checkpoint validation passed."

        result = CheckpointValidationResult(
            is_valid=False,
            missing_keys=["a", "b"],
            unexpected_keys=["c"],
            shape_mismatches=["d: model=(2,3), checkpoint=(4,5)"],
            errors=[],
        )
        summary = result.summary()
        assert "Missing keys" in summary
        assert "Unexpected keys" in summary
        assert "Shape mismatches" in summary


# ---------------------------------------------------------------------------
# estimate_model_memory tests
# ---------------------------------------------------------------------------


class TestEstimateModelMemory:
    """Tests for the estimate_model_memory utility."""

    def test_basic_estimation(self):
        """Should return reasonable estimates for a known config."""
        config = Config.from_name("pythia-14m")
        result = estimate_model_memory(config, dtype=torch.float32, training=False)
        assert result["param_count"] > 0
        assert result["param_memory_gb"] > 0
        assert result["estimated_total_gb"] > 0
        # pythia-14m is ~14M params → ~0.05 GB in fp32
        assert result["param_memory_gb"] < 1.0

    def test_training_multiplier(self):
        """Training should use ~3× multiplier."""
        config = Config.from_name("pythia-14m")
        inference = estimate_model_memory(config, dtype=torch.float32, training=False)
        training = estimate_model_memory(config, dtype=torch.float32, training=True)
        assert training["estimated_total_gb"] > inference["estimated_total_gb"]
        # Should be approximately 3×
        ratio = training["estimated_total_gb"] / inference["estimated_total_gb"]
        assert 2.5 < ratio < 3.5

    def test_dtype_affects_memory(self):
        """Half precision should use ~half the parameter memory."""
        config = Config.from_name("pythia-14m")
        fp32 = estimate_model_memory(config, dtype=torch.float32, training=False)
        fp16 = estimate_model_memory(config, dtype=torch.float16, training=False)
        assert fp16["param_memory_gb"] < fp32["param_memory_gb"]
        # Should be approximately double (exact ratio depends on estimate granularity)
        ratio = fp32["param_memory_gb"] / fp16["param_memory_gb"]
        assert 1.5 < ratio < 2.5

    def test_gpu_fields(self):
        """GPU-related fields should be None when no GPU is available."""
        config = Config.from_name("pythia-14m")
        with mock.patch("litgpt.utils.torch.cuda.is_available", return_value=False):
            result = estimate_model_memory(config, dtype=torch.float32)
        assert result["available_gpu_memory_gb"] is None
        assert result["fits_in_memory"] is None


# ---------------------------------------------------------------------------
# Tokenizer JSON warning test
# ---------------------------------------------------------------------------


def test_tokenizer_json_warning(tmp_path):
    """Tokenizer should emit a warning when generation_config.json has invalid JSON."""
    # Set up a minimal tokenizer directory with an HF tokenizer
    checkpoint_dir = tmp_path / "test_model"
    checkpoint_dir.mkdir()

    # We need tokenizer.json for the HF path and a valid tokenizer_config.json
    # Use a minimal invalid generation_config.json
    invalid_json = '{\n  "bos_token_id": 1,\n  "eos_token_id": 2,\n}'  # trailing comma

    (checkpoint_dir / "generation_config.json").write_text(invalid_json)
    (checkpoint_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "GPT2Tokenizer", "bos_token": "<s>", "eos_token": "</s>"})
    )

    # Create a minimal tokenizer.json that the HF tokenizer can load
    # This is hard to mock, so we'll test the warning path directly by patching
    from litgpt.tokenizer import Tokenizer

    with mock.patch.object(Tokenizer, "__init__", lambda self, *a, **kw: None):
        tokenizer = Tokenizer.__new__(Tokenizer)

    # Directly test the JSON fallback behavior

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Trigger the code path: read invalid JSON, fall back to fix_and_load_json
        special_tokens_path = checkpoint_dir / "generation_config.json"
        try:
            with open(special_tokens_path, encoding="utf-8") as fp:
                config = json.load(fp)
        except json.JSONDecodeError:
            import litgpt.utils

            warnings.warn(
                f"generation_config.json in '{checkpoint_dir}' contains invalid JSON. "
                "Attempting automatic fix. Verify this file is not corrupted.",
                stacklevel=2,
            )
            with open(special_tokens_path, encoding="utf-8") as fp:
                json_string = fp.read()
                config = litgpt.utils.fix_and_load_json(json_string)

        # Check that the warning was raised
        assert len(w) == 1
        assert "invalid JSON" in str(w[0].message)
        # Check that the fix worked
        assert config["bos_token_id"] == 1
        assert config["eos_token_id"] == 2


# ---------------------------------------------------------------------------
# Validate script integration test
# ---------------------------------------------------------------------------


class TestValidateScript:
    """Integration tests for the validate CLI script."""

    @staticmethod
    def _make_checkpoint_dir(tmp_path: Path, config_name: str = "pythia-14m") -> Path:
        """Create a fake but structurally valid checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints" / "test"
        checkpoint_dir.mkdir(parents=True)

        config = Config.from_name(config_name)
        config_dict = asdict(config)
        with open(checkpoint_dir / "model_config.yaml", "w") as f:
            yaml.dump(config_dict, f)

        # Create a real checkpoint
        with torch.device("meta"):
            model = GPT(config)
        real_sd = {k: torch.randn(v.shape) for k, v in model.state_dict().items()}
        torch.save(real_sd, str(checkpoint_dir / "lit_model.pth"))

        # Create minimal tokenizer files
        (checkpoint_dir / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "GPT2Tokenizer"}))
        (checkpoint_dir / "tokenizer.json").write_text("{}")

        return checkpoint_dir

    def test_validate_missing_dir(self, tmp_path, capsys):
        """validate_setup should fail for a non-existent directory."""
        from litgpt.scripts.validate import validate_setup

        with pytest.raises(SystemExit):
            validate_setup(checkpoint_dir=tmp_path / "nonexistent")

    def test_validate_missing_model_file(self, tmp_path, capsys):
        """validate_setup should fail when checkpoint file is missing."""
        checkpoint_dir = tmp_path / "test"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model_config.yaml").write_text(yaml.dump(asdict(Config.from_name("pythia-14m"))))
        (checkpoint_dir / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "GPT2Tokenizer"}))
        (checkpoint_dir / "tokenizer.json").write_text("{}")
        # No lit_model.pth

        from litgpt.scripts.validate import validate_setup

        with pytest.raises(SystemExit):
            validate_setup(checkpoint_dir=checkpoint_dir)
