import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Literal, Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from lm_eval.base import BaseLM

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lm_eval_harness import EvalHarnessBase

from lit_gpt import Tokenizer
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


class EvalHarnessLoRA(EvalHarnessBase):
    def __init__(
        self,
        lora_path: str = "",
        checkpoint_dir: str = "",
        input: str = "",
        precision: str = "bf16-true",
        batch_size=1,
        temperature=1,
        device="auto",
        devices: int = 1,
        strategy: str = "auto",
        quantize: Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"] | None = None,
    ):
        super(BaseLM, self).__init__()
        assert os.path.exists(lora_path)
        assert isinstance(device, str)
        assert isinstance(batch_size, int)
        assert isinstance(checkpoint_dir, str)
        self.input = input
        lora_path = Path(lora_path)
        checkpoint_dir = Path(checkpoint_dir)

        if strategy == "fsdp":
            strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
        fabric = L.Fabric(devices=devices, accelerator=device, precision=precision, strategy=strategy)
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
        self.fabric = fabric
        self.model = fabric.setup(model)
        self.tokenizer = Tokenizer(checkpoint_dir)

        self.batch_size_per_gpu = batch_size
        self.temperature = temperature

    def tok_encode(self, string: str):
        sample = {"instruction": string, "input": self.input}
        prompt = generate_prompt(sample)
        return self.tokenizer.encode(prompt, bos=False, eos=False).tolist()


def run_eval_harness(
    lora_path: str = "",
    checkpoint_dir: str = "",
    input: str = "",
    precision: Optional[str] = None,
    batch_size=1,
    eval_tasks: Optional[List[str]] = None,
    num_fewshot=0,
    bootstrap_iters=2,
    temperature=1.0,
    device="auto",
    devices: int = 1,
    strategy: str = "auto",
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    save_filepath: Optional[str] = None,
):
    precision = precision or get_default_supported_precision(training=False)

    eval_harness = EvalHarnessLoRA(
        lora_path=lora_path,
        checkpoint_dir=checkpoint_dir,
        input=input,
        precision=precision,
        batch_size=batch_size,
        temperature=temperature,
        device=device,
        devices=devices,
        strategy=strategy,
        quantize=quantize,
    )
    eval_harness.fabric.print("Running evaluation harness...")
    results = eval_harness.run_eval(
        eval_tasks=eval_tasks, num_fewshot=num_fewshot, bootstrap_iters=bootstrap_iters, use_cache=False
    )
    if save_filepath:
        data = json.dumps(results)
        with open(save_filepath, "w") as fw:
            fw.write(data)
        print(f"Results saved at {save_filepath}")
    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    result = CLI(run_eval_harness)
    print(result)
