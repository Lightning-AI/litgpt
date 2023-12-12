import itertools
import logging
import re
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, Literal, Optional, Type, Union
from warnings import filterwarnings

import lightning as L
import torch
from lightning.fabric.accelerators import CUDAAccelerator
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.utilities.throughput import _plugin_to_compute_dtype

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import generate.base as generate_base
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import Block, build_mask_cache
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision


@torch.inference_mode()
def get_model(fabric: L.Fabric, config: Config, max_seq_length: int, devices: int):
    root = fabric.device

    with torch.device("meta"):
        model = GPT(config)

    if model.config.n_layer % devices:
        raise NotImplementedError(
            f"Only balanced partitioning is implemented: n_layer={model.config.n_layer}, devices {devices}"
        )
    layers_per_rank = model.config.n_layer // devices
    # dictates where each block should be instantiated
    mapping = layer_to_device(model, chunk_on=Block, chunk_size=layers_per_rank)
    # materialize each block on the appropriate device
    for layer_num, target_index in mapping.items():
        path = f"transformer.h.{layer_num}"
        submodule = model.get_submodule(path)
        target_device = torch.device(root.type, target_index)
        materialize_meta_tensors(submodule, target_device)
        # and build the kv cache
        submodule.attn.kv_cache = submodule.attn.build_kv_cache(1, max_seq_length, model.cos.size(-1), target_device)
    # rebuild odd ends
    with root:
        # the rope cache which is on meta device
        model.max_seq_length = max_seq_length
        # the mask cache which cannot be created with `set_kv_cache` because that will set it for all layers
        model.mask_cache = build_mask_cache(max_seq_length)
    # and everything that is not a block in the root
    materialize_meta_tensors(model, root)

    # quantize
    # FIXME

    if devices > 1:
        # setup hooks to pipeline layers
        for layer_num, target_index in mapping.items():
            path = f"transformer.h.{layer_num}"
            submodule = model.get_submodule(path)
            if layer_num >= layers_per_rank:
                # we need to move the block input on the boundaries between devices
                # and also on every non-root device because the RoPE and mask cache is shared
                # TODO: the second case could be optimized and then we would only need this hook for
                # `layer_num in [layers_per_rank * i - 1 for i in range(1, devices + 1)]`
                target_device = torch.device(root.type, target_index)
                submodule.register_forward_pre_hook(partial(move_block_input, target_device))
            if layer_num == config.n_layer - 1:
                submodule.register_forward_hook(partial(move_block_output, root))

    return model


def layer_to_device(module: torch.nn.Module, chunk_on: Type[torch.nn.Module], chunk_size: int) -> Dict[int, int]:
    """Create a mapping from layer (block) number to device."""
    mapping = {}
    for name, submodule in module.named_modules():
        if isinstance(submodule, chunk_on):
            split = name.split(".")
            number = int(split[2])
            mapping[number] = number // chunk_size
    return mapping


def materialize(module: torch.nn.Module, device: torch.device) -> None:
    """Materialize a module."""
    module.to_empty(device=device, recurse=False)
    module.reset_parameters()


def materialize_meta_tensors(module: torch.nn.Module, device: torch.device) -> None:
    """Materialize all tensors in a given module."""
    for module in module.modules():
        if any(t.is_meta for t in itertools.chain(module.parameters(recurse=False), module.buffers(recurse=False))):
            materialize(module, device)


def move_block_input(device: torch.device, module: torch.nn.Module, ins):
    """``forward_pre_hook`` to move a Block's input before forward."""
    # during inference, none of the inputs are None: x, cos, sin, mask, input_pos
    return tuple(t.to(device) for t in ins)


def move_block_output(device: torch.device, module: torch.nn.Module, ins, outs) -> torch.Tensor:
    """``forward_hook`` to move a Block's output after forward."""
    return outs.to(device)


@torch.inference_mode()
def main(
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/mistralai/Mistral-7B-Instruct-v0.1"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    devices: Union[int, str] = "auto",
    precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None:
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, accelerator="cuda", plugins=plugins)

    if devices == "auto":
        total_devices = CUDAAccelerator.auto_device_count()
    else:
        total_devices = sum(CUDAAccelerator.parse_devices(devices))

    dtype = _plugin_to_compute_dtype(fabric.strategy.precision)
    fabric.print(f"Using {total_devices} devices, {dtype} as compute dtype", file=sys.stderr)
    torch.set_default_dtype(dtype)

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    model = get_model(fabric, config, max_returned_tokens, total_devices)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    model.eval()

    # for this script, this warning is a false-positive
    filterwarnings("ignore", ".*copying from a non-meta parameter.*", module="torch.nn.modules.module")
    t0 = time.perf_counter()
    state_dict = torch.load(str(checkpoint_path), mmap=True)
    model.load_state_dict(state_dict)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    if compile:
        # silence developer warning on nightly builds
        # https://github.com/pytorch/pytorch/blob/v2.2.0-rc5/torch/_inductor/ir.py#L4166
        pattern = re.compile(".*DeviceCopy in input program.*")
        logging.getLogger("torch._inductor.utils").addFilter(lambda record: not pattern.search(record.getMessage()))
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        # cannot use cudagraphs because it doesn't support multiple device indices
        # https://github.com/pytorch/pytorch/blob/v2.2.0-rc5/torch/_inductor/compile_fx.py#L371-L375
        generate_base.next_token = torch.compile(generate_base.next_token)

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate_base.generate(
            model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id
        )
        t = time.perf_counter() - t0
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
    fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
