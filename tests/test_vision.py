# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Tests for litgpt.vision — multimodal support components."""

import pytest
import torch
import torch.nn as nn

from litgpt.config import Config
from litgpt.vision import (
    ImagePreprocessor,
    MultiModalProjector,
    VisionEncoder,
    merge_input_embeds,
)


# ---------------------------------------------------------------------------
# Helper to build a minimal multimodal Config
# ---------------------------------------------------------------------------
def _mm_config(**overrides):
    """Build a minimal Config suitable for multimodal tests."""
    defaults = dict(
        name="test-mm",
        hf_config={"name": "test-mm", "org": "test"},
        block_size=128,
        vocab_size=512,
        n_layer=2,
        n_head=4,
        n_embd=64,
        padding_multiple=64,
        # Vision fields
        vision_feature_dim=32,
        vision_start_token_id=100,
        vision_patch_size=14,
        vision_image_size=28,  # small to keep tests fast
        mm_projector_type="linear",
    )
    defaults.update(overrides)
    return Config(**defaults)


def _text_only_config(**overrides):
    """Build a minimal text-only Config (no vision fields)."""
    defaults = dict(
        name="test-text",
        hf_config={"name": "test-text", "org": "test"},
        block_size=128,
        vocab_size=512,
        n_layer=2,
        n_head=4,
        n_embd=64,
        padding_multiple=64,
    )
    defaults.update(overrides)
    return Config(**defaults)


# ===== Config Tests =====
class TestConfigMultimodal:
    def test_is_multimodal_true(self):
        config = _mm_config()
        assert config.is_multimodal is True

    def test_is_multimodal_false(self):
        config = _text_only_config()
        assert config.is_multimodal is False

    def test_vision_fields_default_to_none(self):
        config = _text_only_config()
        assert config.vision_feature_dim is None
        assert config.vision_start_token_id is None
        assert config.vision_patch_size is None
        assert config.vision_image_size is None
        assert config.mm_projector_type is None
        assert config.vision_model_name is None


# ===== VisionEncoder Tests =====
class TestVisionEncoder:
    def test_fallback_forward_shape(self):
        config = _mm_config()
        encoder = VisionEncoder(config)
        # image_size=28, patch_size=14 -> 2x2 = 4 patches
        pixel_values = torch.randn(1, 3, 28, 28)
        features = encoder(pixel_values)
        assert features.shape == (1, 4, 32)  # (B, num_patches, vision_feature_dim)

    def test_fallback_batched(self):
        config = _mm_config()
        encoder = VisionEncoder(config)
        pixel_values = torch.randn(3, 3, 28, 28)
        features = encoder(pixel_values)
        assert features.shape == (3, 4, 32)

    def test_requires_vision_feature_dim(self):
        config = _text_only_config()
        with pytest.raises(ValueError, match="vision_feature_dim"):
            VisionEncoder(config)

    def test_num_patches_property(self):
        config = _mm_config()
        encoder = VisionEncoder(config)
        assert encoder.num_patches == 4  # (28/14)^2


# ===== MultiModalProjector Tests =====
class TestMultiModalProjector:
    def test_linear_projector_shape(self):
        proj = MultiModalProjector(vision_dim=32, text_dim=64, projector_type="linear")
        x = torch.randn(2, 4, 32)
        out = proj(x)
        assert out.shape == (2, 4, 64)

    def test_mlp2x_projector_shape(self):
        proj = MultiModalProjector(vision_dim=32, text_dim=64, projector_type="mlp2x")
        x = torch.randn(2, 4, 32)
        out = proj(x)
        assert out.shape == (2, 4, 64)

    def test_invalid_projector_type(self):
        with pytest.raises(ValueError, match="Unknown projector type"):
            MultiModalProjector(vision_dim=32, text_dim=64, projector_type="bad")

    def test_linear_is_single_layer(self):
        proj = MultiModalProjector(vision_dim=32, text_dim=64, projector_type="linear")
        assert isinstance(proj.proj, nn.Linear)

    def test_mlp2x_is_sequential(self):
        proj = MultiModalProjector(vision_dim=32, text_dim=64, projector_type="mlp2x")
        assert isinstance(proj.proj, nn.Sequential)
        assert len(proj.proj) == 3  # Linear + GELU + Linear


# ===== merge_input_embeds Tests =====
class TestMergeInputEmbeds:
    def test_basic_merge(self):
        B, T, D = 1, 10, 64
        N_patches = 4
        image_token_id = 999

        # Create input_ids with 4 image placeholders at positions 2,3,4,5
        input_ids = torch.arange(T).unsqueeze(0)  # (1, 10)
        input_ids[0, 2:6] = image_token_id

        text_embeds = torch.zeros(B, T, D)
        image_embeds = torch.ones(B, N_patches, D)

        merged = merge_input_embeds(text_embeds, image_embeds, image_token_id, input_ids)
        assert merged.shape == (B, T, D)

        # Positions 2-5 should have image embeddings (all ones)
        assert torch.allclose(merged[0, 2:6], torch.ones(N_patches, D))
        # Other positions should still be zeros
        assert torch.allclose(merged[0, :2], torch.zeros(2, D))
        assert torch.allclose(merged[0, 6:], torch.zeros(4, D))

    def test_batch_merge(self):
        B, T, D = 2, 8, 32
        N_patches = 2
        image_token_id = 999

        input_ids = torch.zeros(B, T, dtype=torch.long)
        input_ids[0, 1:3] = image_token_id
        input_ids[1, 5:7] = image_token_id

        text_embeds = torch.zeros(B, T, D)
        image_embeds = torch.ones(B, N_patches, D) * 2.0

        merged = merge_input_embeds(text_embeds, image_embeds, image_token_id, input_ids)
        assert merged.shape == (B, T, D)

        # Check first batch element
        assert merged[0, 1, 0].item() == 2.0
        assert merged[0, 2, 0].item() == 2.0
        assert merged[0, 0, 0].item() == 0.0

        # Check second batch element
        assert merged[1, 5, 0].item() == 2.0
        assert merged[1, 6, 0].item() == 2.0
        assert merged[1, 0, 0].item() == 0.0

    def test_wrong_placeholder_count_raises(self):
        B, T, D = 1, 8, 32
        N_patches = 4
        image_token_id = 999

        input_ids = torch.zeros(B, T, dtype=torch.long)
        input_ids[0, 0:2] = image_token_id  # Only 2, not 4

        text_embeds = torch.zeros(B, T, D)
        image_embeds = torch.ones(B, N_patches, D)

        with pytest.raises(ValueError, match="Expected 4"):
            merge_input_embeds(text_embeds, image_embeds, image_token_id, input_ids)

    def test_does_not_modify_original(self):
        B, T, D = 1, 6, 16
        N_patches = 2
        image_token_id = 999

        input_ids = torch.zeros(B, T, dtype=torch.long)
        input_ids[0, 1:3] = image_token_id

        text_embeds = torch.zeros(B, T, D)
        image_embeds = torch.ones(B, N_patches, D)

        original = text_embeds.clone()
        merge_input_embeds(text_embeds, image_embeds, image_token_id, input_ids)
        assert torch.allclose(text_embeds, original), "Original embeddings should not be modified"


# ===== GPT Model Integration Tests =====
class TestGPTMultimodal:
    def test_text_only_model_unchanged(self):
        """Text-only models should work exactly as before."""
        config = _text_only_config()
        from litgpt.model import GPT

        model = GPT(config)
        assert model.vision_encoder is None
        assert model.mm_projector is None

        # Forward should work without pixel_values
        idx = torch.randint(0, config.padded_vocab_size, (1, 5))
        output = model(idx)
        assert output.shape == (1, 5, config.padded_vocab_size)

    def test_multimodal_model_has_vision_components(self):
        config = _mm_config()
        from litgpt.model import GPT

        model = GPT(config)
        assert model.vision_encoder is not None
        assert model.mm_projector is not None

    def test_multimodal_forward_with_pixel_values(self):
        """Test that forward works when pixel_values are provided."""
        config = _mm_config()
        from litgpt.model import GPT

        model = GPT(config)
        model.eval()

        # Create input with image placeholders
        N_patches = 4  # (28/14)^2
        T = 10
        idx = torch.randint(0, config.padded_vocab_size, (1, T))
        # Place image tokens at positions 2-5
        idx[0, 2:6] = config.vision_start_token_id

        pixel_values = torch.randn(1, 3, 28, 28)

        with torch.no_grad():
            output = model(idx, pixel_values=pixel_values)

        assert output.shape == (1, T, config.padded_vocab_size)

    def test_multimodal_forward_without_pixel_values(self):
        """Multimodal model should still work for text-only inference."""
        config = _mm_config()
        from litgpt.model import GPT

        model = GPT(config)
        model.eval()

        idx = torch.randint(0, config.padded_vocab_size, (1, 5))

        with torch.no_grad():
            output = model(idx)

        assert output.shape == (1, 5, config.padded_vocab_size)


# ===== ImagePreprocessor Tests =====
class TestImagePreprocessor:
    def test_output_shape(self, tmp_path):
        """Test that preprocessor produces correct tensor shape."""
        # Create a dummy image file
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (100, 100), color=(128, 64, 32))
        img_path = tmp_path / "test.jpg"
        img.save(str(img_path))

        preprocessor = ImagePreprocessor(image_size=28)
        result = preprocessor(str(img_path))

        assert result.shape == (1, 3, 28, 28)
        assert result.dtype == torch.float32

    def test_pil_image_input(self):
        """Test that preprocessor accepts PIL Image objects."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (50, 50), color=(255, 0, 0))
        preprocessor = ImagePreprocessor(image_size=14)
        result = preprocessor(img)
        assert result.shape == (1, 3, 14, 14)

    def test_normalization(self, tmp_path):
        """Test that output is normalized (not in [0, 255] range)."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (28, 28), color=(128, 128, 128))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        preprocessor = ImagePreprocessor(image_size=28)
        result = preprocessor(str(img_path))

        # After normalization, values should not be in raw [0, 255] range
        assert result.max().item() < 10.0
        assert result.min().item() > -10.0
