import sys
from pathlib import Path
from typing import Optional

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def download_from_hub(repo_id: Optional[str] = None, directory: Optional[str] = None) -> str:
    if directory is None:
        directory = "checkpoints/"

    if repo_id is None:
        from lit_gpt.config import configs

        options = [f"{config['org']}/{config['name']}" for config in configs]
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(options))
        return ""

    from huggingface_hub import snapshot_download

    checkpoint_dir: str = str(Path(directory) / Path(repo_id))

    snapshot_download(
        repo_id,
        local_dir=checkpoint_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.bin*", "tokenizer*"],
    )
    return checkpoint_dir


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)
