# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
#
# Vision encoder and multimodal projection utilities for VLMs.

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from litgpt.config import Config


class VisionEncoder(nn.Module):
    """Wraps a pretrained vision backbone (CLIP, SigLIP, etc.) and extracts patch features.

    The encoder is kept **frozen** by default – only the projector is trainable.

    Args:
        config: The LitGPT model config (must have ``vision_feature_dim`` set).
        pretrained_model_name: Optional HuggingFace model name. If provided,
            loads weights from HuggingFace ``transformers``.
    """

    def __init__(self, config: Config, pretrained_model_name: str | None = None) -> None:
        super().__init__()
        if config.vision_feature_dim is None:
            raise ValueError("VisionEncoder requires config.vision_feature_dim to be set.")

        self.config = config
        self.vision_feature_dim = config.vision_feature_dim
        self._encoder: nn.Module | None = None

        if pretrained_model_name is not None:
            self._load_hf_encoder(pretrained_model_name)
        else:
            # Placeholder linear for testing / when loading weights separately
            image_size = config.vision_image_size or 224
            patch_size = config.vision_patch_size or 14
            num_patches = (image_size // patch_size) ** 2
            # Simple conv-based patch embedding as fallback
            self.patch_embed = nn.Conv2d(
                3,
                self.vision_feature_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
            self._num_patches = num_patches

    def _load_hf_encoder(self, model_name: str) -> None:
        """Load a vision encoder from HuggingFace transformers."""
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "Loading a pretrained vision encoder requires `transformers`. Install it with: pip install transformers"
            )
        self._encoder = AutoModel.from_pretrained(model_name)
        # Freeze the vision encoder
        for param in self._encoder.parameters():
            param.requires_grad = False

    @property
    def num_patches(self) -> int:
        """Number of image patch tokens produced per image."""
        if self._encoder is not None:
            # Try to get from config
            image_size = self.config.vision_image_size or 224
            patch_size = self.config.vision_patch_size or 14
            return (image_size // patch_size) ** 2
        return self._num_patches

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: ``(B, C, H, W)`` image tensor, pre-normalized.

        Returns:
            Image features of shape ``(B, num_patches, vision_feature_dim)``.
        """
        if self._encoder is not None:
            outputs = self._encoder(pixel_values=pixel_values)
            # Most HF vision models return .last_hidden_state
            # Skip the [CLS] token if present
            features = outputs.last_hidden_state
            if features.size(1) == self.num_patches + 1:
                features = features[:, 1:, :]  # remove CLS
            return features
        else:
            # Fallback: simple conv patch embedding
            # pixel_values: (B, 3, H, W)
            x = self.patch_embed(pixel_values)  # (B, D, H', W')
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
            return x


class MultiModalProjector(nn.Module):
    """Maps vision encoder features to the LLM's embedding dimension.

    Supports two projector types:
    - ``"linear"``: Single linear layer.
    - ``"mlp2x"``: Two-layer MLP with GELU activation (LLaVA-style).

    Args:
        vision_dim: Dimension of the vision encoder output.
        text_dim: Dimension of the LLM's token embeddings (``config.n_embd``).
        projector_type: ``"linear"`` or ``"mlp2x"``.
    """

    def __init__(self, vision_dim: int, text_dim: int, projector_type: str = "linear") -> None:
        super().__init__()
        self.projector_type = projector_type

        if projector_type == "linear":
            self.proj = nn.Linear(vision_dim, text_dim, bias=True)
        elif projector_type == "mlp2x":
            self.proj = nn.Sequential(
                nn.Linear(vision_dim, text_dim, bias=True),
                nn.GELU(),
                nn.Linear(text_dim, text_dim, bias=True),
            )
        else:
            raise ValueError(f"Unknown projector type: {projector_type!r}. Supported: 'linear', 'mlp2x'.")

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: ``(B, num_patches, vision_dim)``

        Returns:
            Projected features of shape ``(B, num_patches, text_dim)``.
        """
        return self.proj(image_features)


def merge_input_embeds(
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    image_token_id: int,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Replace ``<image>`` placeholder embeddings with actual image embeddings.

    This function takes the standard text embedding output from ``wte`` and
    splices in projected image patch embeddings at the positions where the
    input contains the ``image_token_id`` placeholder token.

    Args:
        text_embeds: ``(B, T, D)`` – embeddings from ``model.transformer.wte(idx)``.
        image_embeds: ``(B, N_patches, D)`` – projected image patch embeddings.
        image_token_id: The token ID used as the ``<image>`` placeholder.
        input_ids: ``(B, T)`` – the original token IDs (needed to locate placeholders).

    Returns:
        Merged embeddings ``(B, T, D)`` with image patches replacing placeholders.

    Raises:
        ValueError: If the number of ``<image>`` placeholders doesn't match
            the number of image patches.
    """
    B, T, D = text_embeds.shape
    N_patches = image_embeds.size(1)

    # Find positions of image placeholder tokens
    image_mask = input_ids == image_token_id  # (B, T)

    # Validate: each batch element should have exactly N_patches placeholders
    counts = image_mask.sum(dim=1)  # (B,)
    if not (counts == N_patches).all():
        raise ValueError(
            f"Expected {N_patches} <image> placeholder tokens per sequence, but got counts: {counts.tolist()}"
        )

    # Clone so we don't modify the original
    merged = text_embeds.clone()

    # For each batch element, scatter image embeddings
    for b in range(B):
        positions = image_mask[b].nonzero(as_tuple=True)[0]  # (N_patches,)
        merged[b, positions] = image_embeds[b]

    return merged


class ImagePreprocessor:
    """Handles image loading, resizing, and normalization for VLMs.

    This class loads images from file paths or PIL Image objects, resizes
    them to the expected input size, and normalizes pixel values.

    Args:
        image_size: Target image size (both height and width).
        mean: Per-channel normalization mean (default: ImageNet).
        std: Per-channel normalization std (default: ImageNet).
    """

    # ImageNet defaults (used by CLIP, SigLIP, etc.)
    IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        image_size: int = 224,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
    ) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(
        self,
        image: str | Path | Any,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Preprocess an image into a normalized tensor.

        Args:
            image: A file path (str/Path) or a PIL Image object.
            device: Target device for the output tensor.

        Returns:
            ``(1, 3, image_size, image_size)`` tensor, normalized.
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError("Image preprocessing requires Pillow. Install it with: pip install Pillow")

        if isinstance(image, (str, Path)):
            img = PILImage.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        # Resize with bicubic interpolation
        img = img.resize((self.image_size, self.image_size), PILImage.BICUBIC)

        # Convert to tensor: (H, W, C) -> (C, H, W), scale to [0, 1]
        import numpy as np

        pixel_values = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        # Normalize
        mean = torch.tensor(self.mean, dtype=pixel_values.dtype).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=pixel_values.dtype).view(3, 1, 1)
        pixel_values = (pixel_values - mean) / std

        # Add batch dimension and move to device
        return pixel_values.unsqueeze(0).to(device)
