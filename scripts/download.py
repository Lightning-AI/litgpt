import os
import sys
from pathlib import Path
from typing import Optional

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def download_from_hub(repo_id: Optional[str] = None, token: Optional[str] = os.getenv("HF_TOKEN")) -> None:
    if repo_id is None:
        from lit_gpt.config import configs

        options = [f"{config['org']}/{config['name']}" for config in configs]
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(options))
        return

    from huggingface_hub import snapshot_download

    if "meta-llama" in repo_id and not token:
        raise ValueError(
            "the meta-llama models require authentication, please set the `HF_TOKEN=your_token` environment"
            " variable or pass --token=your_token. You can find your token by visiting"
            " https://huggingface.co/settings/tokens"
        )

    snapshot_download(
        repo_id,
        local_dir=f"checkpoints/{repo_id}",
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.bin*", "tokenizer*", "generation_config.json"],
        token=token,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)
