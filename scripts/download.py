import os
import sys
from pathlib import Path
from typing import Optional
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def download_from_hub(
        repo_id: Optional[str] = None,
        access_token: Optional[str] = os.getenv("HF_TOKEN"),
        from_safetensors: Optional[bool] = False,
        ) -> None:
    if repo_id is None:
        from lit_gpt.config import configs

        options = [f"{config['org']}/{config['name']}" for config in configs]
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(options))
        return

    from huggingface_hub import snapshot_download

    if "meta-llama" in repo_id and not access_token:
        raise ValueError(
            "the meta-llama models require authentication, please set the `HF_TOKEN=your_token` environment"
            " variable or pass --access_token=your_token. You can find your token by visiting"
            " https://huggingface.co/settings/tokens"
        )

    download_files = ["tokenizer*", "generation_config.json"]
    if from_safetensors:
        try:
            import safetensors
        except ImportError:
            print("safetensors is not installed. Install safetensors"
                  " (`pip install safetensors`) and run this script again.")
            quit()
        download_files.extend("*.safetensors*")
    else:
        download_files.extend("*.bin*")

    snapshot_download(
        repo_id,
        local_dir=f"checkpoints/{repo_id}",
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=download_files,
        token=access_token,
    )

    # convert safetensors to PyTorch binaries
    if from_safetensors:
        from safetensors.torch import load_file as safetensors_load

        print("Converting .safetensor files to PyTorch binaries (.bin)")
        directory = f"checkpoints/{repo_id}"
        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)

            if ".safetensors" in filename:
                new_filename = filename.replace('.safetensors', '.bin')
                new_filename = "pytorch_" + new_filename
                new_full_path = os.path.join(directory, new_filename)

                print(f"{filename} --> {new_filename}")
                pt_state_dict = safetensors_load(full_path)
                torch.save(pt_state_dict, new_full_path)
                os.remove(full_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)
