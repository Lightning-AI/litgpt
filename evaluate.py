# This mimics GPTQ's evaluation metrics: https://github.com/IST-DASLab/gptq/
# Thanks to E. Frantar et al GPTQ: Accurate Post-training Compression for GPT, arXiv:2210.17323
import math
import sys
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import tqdm

from lit_llama import LLaMA, Tokenizer, as_8_bit_quantized
from lit_llama.utils import EmptyInitOnDevice

from datasets import load_dataset


def load_eval_data(dataset_name: str) -> str:
    # this mimics gptq datautils
    if dataset_name == "wikitext":
        # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testdata = "\n\n".join(testdata["text"])
    elif dataset_name == "ptb":
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testdata = "\n\n".join(testdata["sentence"])
    elif dataset_name == "c4":
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testdata = " ".join(testdata[:1100]["text"])

    else:
        raise ValueError("invalid dataset name (wikitext, ptb, c4 are allowed)")
    return testdata


def main(
    datasets: str = "wikitext,ptb,c4",
    *,
    # compilation fails as it does not support torch.complex64 for RoPE
    # compile: bool = False,
    accelerator: str = "auto",
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    model_size: str = "7B",
    dtype: Optional[str] = None,
    quantize: bool = False,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        datasets: The datasets to use as a comma separated string
        # compile: Whether to compile the model.
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model using the `LLM.int8()` method
    """
    if not checkpoint_path:
        checkpoint_path = Path(f"./checkpoints/lit-llama/{model_size}/state_dict.pth")
    if not tokenizer_path:
        tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    fabric = L.Fabric(accelerator=accelerator, devices=1)

    quantization_mode = "llm.int8" if quantize else None
    if dtype is not None:
        dt = getattr(torch, dtype, None)
        if not isinstance(dt, torch.dtype):
            raise ValueError(f"{dtype} is not a valid dtype.")
        dtype = dt

    with EmptyInitOnDevice(
        device=fabric.device, dtype=dtype, quantization_mode=quantization_mode
    ):
        print("Loading model ...", file=sys.stderr)
        t0 = time.time()
        model = LLaMA.from_name(model_size)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()

    # if compile:
    #     model = torch.compile(model)

    total_toks = 0
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)

    for dsname in datasets.split(","):
        test_string = load_eval_data(dsname)
        encoded_text = tokenizer.encode(
            test_string, bos=True, eos=False, device=fabric.device
        )
        encoded_text = encoded_text[
            None, : 256 * model.config.block_size
        ]  # add batch dimension, trim like gptq implementation
        t0 = time.perf_counter()

        nlls = 0
        toks = 0
        with torch.inference_mode():
            block_size = 2048  # this is for compat with gptq, and indeed we get much worse beyond this (https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L30)
            for i in tqdm.tqdm(range(0, encoded_text.shape[1], block_size)):
                inp = encoded_text[:, i : i + block_size]
                logits = model(inp)[0]
                nll = torch.nn.functional.cross_entropy(
                    logits[:-1], inp[0, 1:].to(dtype=torch.long), reduction="sum"
                )
                toks += inp.size(1) - 1
                nlls += nll.item()

        print(encoded_text.shape, logits.shape)
        encoded_text = encoded_text[:, : logits.shape[0]]
        ppl = math.exp(nlls / toks)
        print(f"Perplexity on {dsname}: {ppl:.2f}")
        total_toks += toks

    t = time.perf_counter() - t0
    print(
        f"\n\nTime for inference: {t:.02f} sec total, {total_toks / t:.02f} tokens/sec",
        file=sys.stderr,
    )
    print(
        f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
