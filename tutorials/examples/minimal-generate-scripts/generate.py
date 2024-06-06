# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from pathlib import Path
import torch
from litgpt.generate.base import main
from litgpt.utils import get_default_supported_precision


def use_model():

    # run `litgpt download EleutherAI/pythia-1b` to download the checkpoint first
    checkpoint_dir = Path("checkpoints") / "EleutherAI" / "pythia-1b"

    torch.manual_seed(123)

    main(
        prompt="What food do llamas eat?",
        max_new_tokens=50,
        temperature=0.5,
        top_k=200,
        top_p=1.0,
        checkpoint_dir=checkpoint_dir,
        precision=get_default_supported_precision(training=False),
        compile=False
    )


if __name__ == "__main__":
    use_model()
