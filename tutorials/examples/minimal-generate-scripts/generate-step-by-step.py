# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from pathlib import Path

import lightning as L
import torch

from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer
from litgpt.utils import load_checkpoint, get_default_supported_precision
from litgpt.generate.base import generate
from litgpt.model import GPT
from litgpt.config import Config


def use_model():

    ###################
    # Load model
    ###################

    # run `litgpt download EleutherAI/pythia-1b` to download the checkpoint first
    checkpoint_dir = Path("checkpoints") / "EleutherAI" / "pythia-1b"
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    precision = get_default_supported_precision(training=False)
    device = torch.device("cuda")

    fabric = L.Fabric(
        accelerator=device.type,
        devices=1,
        precision=precision,
    )

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    tokenizer = Tokenizer(checkpoint_dir)

    prompt_style = PromptStyle.from_config(config)

    with fabric.init_module(empty_init=True):
        model = GPT(config)
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)

    model.eval()
    model = fabric.setup_module(model)
    load_checkpoint(fabric, model, checkpoint_path)

    device = fabric.device

    ###################
    # Predict
    ###################

    prompt = "What do Llamas eat?"
    max_new_tokens = 50

    prompt = prompt_style.apply(prompt)
    encoded = tokenizer.encode(prompt, device=device)

    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    torch.manual_seed(123)

    y = generate(
        model,
        encoded,
        max_returned_tokens,
        temperature=0.5,
        top_k=200,
        top_p=1.0,
        eos_id=tokenizer.eos_id
    )

    for block in model.transformer.h:
        block.attn.kv_cache.reset_parameters()

    decoded_output = tokenizer.decode(y)
    print(decoded_output)


if __name__ == "__main__":
    use_model()