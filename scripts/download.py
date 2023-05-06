def download_from_hub(repo_id: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, local_dir=f"checkpoints/{repo_id}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)
