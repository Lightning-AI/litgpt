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

    With ``pretrained_model_name``, the HuggingFace architecture is built from its config only, so
    the model can be created on the meta device. Pretrained weights are loaded right away when the
    module is created on a real device, and otherwise when a checkpoint without encoder weights is loaded.

    Args:
        config: The LitGPT model config (must have ``vision_feature_dim`` set).
        pretrained_model_name: Optional HuggingFace model name for the vision backbone.
    """

    def __init__(self, config: Config, pretrained_model_name: str | None = None) -> None:
        super().__init__()
        if config.vision_feature_dim is None:
            raise ValueError("VisionEncoder requires config.vision_feature_dim to be set.")

        self.config = config
        self.vision_feature_dim = config.vision_feature_dim
        self.pretrained_model_name = pretrained_model_name
        self._encoder: nn.Module | None = None

        if pretrained_model_name is not None:
            self._build_hf_encoder(pretrained_model_name)
            if not any(p.is_meta for p in self._encoder.parameters()):
                self.load_pretrained_weights()
            self._register_load_state_dict_pre_hook(self._fill_missing_encoder_weights)
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

    def _build_hf_encoder(self, model_name: str) -> None:
        """Build the vision tower of a HuggingFace model from its config, without loading weights."""
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError:
            raise ImportError(
                "Loading a pretrained vision encoder requires `transformers`. Install it with: pip install transformers"
            )
        hf_config = AutoConfig.from_pretrained(model_name)
        # CLIP/SigLIP configs describe dual-tower models whose forward also needs `input_ids`;
        # keep only the vision tower.
        model = AutoModel.from_config(getattr(hf_config, "vision_config", hf_config))
        self._encoder = getattr(model, "vision_model", model)
        # Freeze the vision encoder
        for param in self._encoder.parameters():
            param.requires_grad = False

    def _pretrained_state_dict(self) -> dict[str, torch.Tensor]:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(self.pretrained_model_name)
        return getattr(model, "vision_model", model).state_dict()

    def load_pretrained_weights(self) -> None:
        """Load the HuggingFace pretrained weights into the vision tower."""
        self._encoder.load_state_dict(self._pretrained_state_dict())

    def _fill_missing_encoder_weights(self, state_dict: dict[str, Any], prefix: str, *args: Any) -> None:
        # LitGPT checkpoints converted from text-only weights carry no vision tower; take it from HF.
        encoder_prefix = f"{prefix}_encoder."
        if any(k.startswith(encoder_prefix) for k in state_dict):
            return
        for k, v in self._pretrained_state_dict().items():
            state_dict[encoder_prefix + k] = v

    @property
    def num_patches(self) -> int:
        """Number of image patch tokens produced per image."""
        if self._encoder is not None:
            encoder_config = getattr(self._encoder, "config", None)
            image_size = getattr(encoder_config, "image_size", None) or self.config.vision_image_size or 224
            patch_size = getattr(encoder_config, "patch_size", None) or self.config.vision_patch_size or 14
            if isinstance(image_size, (tuple, list)):
                image_size = image_size[0]
            if isinstance(patch_size, (tuple, list)):
                patch_size = patch_size[0]
            return (image_size // patch_size) ** 2
        return self._num_patches

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: ``(B, C, H, W)`` image tensor, pre-normalized.

        Returns:
            Image features of shape ``(B, num_patches, vision_feature_dim)``.
        """
        encoder = self._encoder if self._encoder is not None else self.patch_embed
        pixel_values = pixel_values.to(dtype=next(encoder.parameters()).dtype)

        if self._encoder is not None:
            # Frozen HF encoder: no gradients needed.
            with torch.no_grad():
                outputs = self._encoder(pixel_values=pixel_values)
                # Most HF vision models return .last_hidden_state
                # Skip the [CLS] token if present
                features = outputs.last_hidden_state
                if features.size(1) == self.num_patches + 1:
                    features = features[:, 1:, :]  # remove CLS
                elif features.size(1) != self.num_patches:
                    raise ValueError(
                        f"Vision encoder returned {features.size(1)} tokens, but expected "
                        f"{self.num_patches} patch tokens (or {self.num_patches + 1} including CLS)."
                    )
                if features.size(-1) != self.vision_feature_dim:
                    raise ValueError(
                        f"Vision encoder returned feature dimension {features.size(-1)}, but config expects "
                        f"{self.vision_feature_dim}."
                    )
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


def expand_image_tokens(
    input_ids: torch.Tensor,
    image_token_id: int,
    num_patches: int,
    bos_id: int | None = None,
) -> torch.Tensor:
    """Make a 1D prompt carry exactly ``num_patches`` ``<image>`` placeholder tokens.

    A single placeholder is expanded in place to ``num_patches`` copies. If the prompt has no
    placeholder, the block is inserted at the start (after ``bos_id`` if the prompt begins with it).
    A prompt that already has ``num_patches`` placeholders is returned unchanged.

    Args:
        input_ids: ``(T,)`` encoded prompt.
        image_token_id: The token ID used as the ``<image>`` placeholder.
        num_patches: Number of image patch embeddings the vision encoder produces.
        bos_id: Optional BOS token ID; the image block is placed after it.

    Returns:
        ``(T')`` token IDs with ``num_patches`` placeholders.
    """
    positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    count = positions.numel()
    if count == num_patches:
        return input_ids
    block = torch.full((num_patches,), image_token_id, dtype=input_ids.dtype, device=input_ids.device)
    if count == 0:
        pos = 1 if bos_id is not None and input_ids.numel() > 0 and input_ids[0].item() == bos_id else 0
    elif count == 1:
        pos = positions.item()
        input_ids = torch.cat((input_ids[:pos], input_ids[pos + 1 :]))
    else:
        raise ValueError(
            f"Prompt contains {count} <image> placeholder tokens; expected 0, 1 or {num_patches} (one image)."
        )
    return torch.cat((input_ids[:pos], block, input_ids[pos:]))


class ImagePreprocessor:
    """Handles image loading, resizing, and normalization for VLMs.

    This class loads images from file paths or PIL Image objects, resizes
    them to the expected input size, and normalizes pixel values.

    Args:
        image_size: Target image size (both height and width).
        mean: Per-channel normalization mean (default: OpenAI CLIP).
        std: Per-channel normalization std (default: OpenAI CLIP).
    """

    # OpenAI CLIP normalization. SigLIP uses 0.5 for every channel; ``from_config`` picks the
    # values that match the configured vision backbone.
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        image_size: int = 224,
        mean: tuple[float, ...] = CLIP_MEAN,
        std: tuple[float, ...] = CLIP_STD,
    ) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std

    @classmethod
    def from_config(cls, config: Config) -> ImagePreprocessor:
        """Build a preprocessor matching the model's vision backbone.

        Uses the HuggingFace image processor's size and normalization when ``config.vision_model_name``
        is set, otherwise ``config.vision_image_size`` with CLIP normalization.
        """
        image_size = config.vision_image_size or 224
        if config.vision_model_name is None:
            return cls(image_size=image_size)
        from transformers import AutoImageProcessor

        hf_processor = AutoImageProcessor.from_pretrained(config.vision_model_name)
        size = getattr(hf_processor, "crop_size", None) or getattr(hf_processor, "size", None) or {}
        if isinstance(size, int):
            image_size = size
        else:
            if not isinstance(size, dict):
                size = vars(size)
            image_size = size.get("height") or size.get("shortest_edge") or image_size
        return cls(
            image_size=image_size,
            mean=tuple(getattr(hf_processor, "image_mean", None) or cls.CLIP_MEAN),
            std=tuple(getattr(hf_processor, "image_std", None) or cls.CLIP_STD),
        )

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
