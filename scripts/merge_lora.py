"""This script merges the LoRA weights with the base model"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.lora import GPT, Block, Config, merge_lora_weights
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load, quantization
from scripts.prepare_alpaca import generate_prompt


lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False


def save_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model})


def main(
    lora_path: Path = Path("out/lora/alpaca/lit_model_lora_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    out_dir: Path = Path("out/lora/checkpoint"),
    strategy: str = "auto",
    devices: int = 1,
    precision: Optional[str] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        out_dir: The path to the merged model that is created by this script.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config_params = dict(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            to_query=lora_query,
            to_key=lora_key,
            to_value=lora_value,
            to_projection=lora_projection,
            to_mlp=lora_mlp,
            to_head=lora_head,
        )
        config_params.update(**json.load(fp))
        del config_params['condense_ratio'] ## TODO: remove and update
        config = Config(**config_params)

    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), quantization(quantize):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(lora_path) as lora_checkpoint:
        checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
        model.load_state_dict(checkpoint, strict=quantize is None)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    merge_lora_weights(model)
    model = fabric.setup(model)

    model.reset_cache()

    # Save the final checkpoint at the end of training
    save_path = out_dir / "lit_model.pth"
    save_checkpoint(fabric, model, save_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    CLI(main)