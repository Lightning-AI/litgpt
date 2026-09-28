# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Download and convert the tutorial model during the image build (no GPU needed)."""

from pathlib import Path

from huggingface_hub import snapshot_download

from litgpt.scripts.convert_hf_checkpoint import convert_hf_checkpoint

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def main() -> None:
    checkpoint = Path("checkpoints") / MODEL
    snapshot_download(
        repo_id=MODEL,
        revision=REVISION,
        local_dir=checkpoint,
        allow_patterns=[
            "*.safetensors",
            "tokenizer*",
            "config.json",
            "generation_config.json",
            "LICENSE*",
            "README.md",
        ],
    )
    convert_hf_checkpoint(checkpoint_dir=checkpoint, model_name="SmolLM2-135M-Instruct", dtype="bfloat16")


if __name__ == "__main__":
    main()
