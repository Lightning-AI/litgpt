from typing import Optional


def download_from_hub(repo_id: Optional[str] = None, local_dir: str = "checkpoints/hf-llama/7B") -> None:
    if repo_id is None:
        raise ValueError("Please pass `--repo_id=...`. You can try googling 'huggingface hub llama' for options.")

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, local_dir=local_dir)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)
