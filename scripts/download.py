# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from lightning_utilities.core.imports import RequirementCache

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")


def download_from_hub(
    repo_id: Optional[str] = None,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    from_safetensors: bool = False,
    tokenizer_only: bool = False,
    checkpoint_dir: Path = Path("checkpoints"),
) -> None:
    if repo_id is None:
        from lit_gpt.config import configs

        options = [f"{config['hf_config']['org']}/{config['hf_config']['name']}" for config in configs]
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(options))
        return

    from huggingface_hub import snapshot_download

    if ("meta-llama" in repo_id or "falcon-180" in repo_id) and not access_token:
        raise ValueError(
            f"{repo_id} requires authentication, please set the `HF_TOKEN=your_token` environment"
            " variable or pass --access_token=your_token. You can find your token by visiting"
            " https://huggingface.co/settings/tokens"
        )

    download_files = ["tokenizer*", "generation_config.json"]
    if not tokenizer_only:
        if from_safetensors:
            if not _SAFETENSORS_AVAILABLE:
                raise ModuleNotFoundError(str(_SAFETENSORS_AVAILABLE))
            download_files.append("*.safetensors")
        else:
            # covers `.bin` files and `.bin.index.json`
            download_files.append("*.bin*")
    elif from_safetensors:
        raise ValueError("`--from_safetensors=True` won't have an effect with `--tokenizer_only=True`")

    directory = checkpoint_dir / repo_id
    snapshot_download(
        repo_id,
        local_dir=directory,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=download_files,
        token=access_token,
    )

    # convert safetensors to PyTorch binaries
    if from_safetensors:
        from safetensors import SafetensorError
        from safetensors.torch import load_file as safetensors_load

        print("Converting .safetensor files to PyTorch binaries (.bin)")
        for safetensor_path in directory.glob("*.safetensors"):
            bin_path = safetensor_path.with_suffix(".bin")
            try:
                result = safetensors_load(safetensor_path)
            except SafetensorError as e:
                raise RuntimeError(f"{safetensor_path} is likely corrupted. Please try to re-download it.") from e
            print(f"{safetensor_path} --> {bin_path}")
            torch.save(result, bin_path)
            os.remove(safetensor_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)
