import re
from pathlib import Path
import sys
from typing import Optional

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def download_from_hub(repo_id: Optional[str] = None) -> None:
    if repo_id is None:
        from lit_gpt.config import configs

        orgs = {
            "stablelm": "stabilityai",
            "pythia": "EleutherAI",
            "RedPajama": "togethercomputer",
            "falcon": "tiiuae",
            "open": "openlm-research",
        }
        options = [f"{orgs[re.split('-|_', name)[0]]}/{name}" for name in configs]
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(options))
        return

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, local_dir=f"checkpoints/{repo_id}", local_dir_use_symlinks=False, resume_download=True)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)
