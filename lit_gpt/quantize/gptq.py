# This adapts GPTQ's quantization process: https://github.com/IST-DASLab/gptq/
# E. Frantar et al GPTQ: Accurate Post-training Compression for GPT, arXiv:2210.17323
# portions copyright by the authors licensed under the Apache License 2.0
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

from datasets import load_dataset
import torch
from lightning import Fabric

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Tokenizer, Config
from lit_gpt.utils import check_valid_checkpoint_dir, lazy_load


class GPTQQuantizer:
    # The algorithm and code has been taken from  https://github.com/IST-DASLab/gptq/
    # E. Frantar et al GPTQ: Accurate Post-training Compression for GPT, arXiv:2210.17323
    # portions copyright by the authors licensed under the Apache License 2.0
    # All errors are our own.

    def __init__(
        self,
        linear_module,
        *,
        bits,
        perchannel=True,
        sym=False,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
    ):
        assert isinstance(linear_module, torch.nn.Linear)

        self.linear_module = linear_module
        self.dev = self.linear_module.weight.device
        self.rows = linear_module.weight.shape[0]
        self.columns = linear_module.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.bits = bits
        self.maxq = 2**bits - 1
        self.perchannel = perchannel
        self.sym = sym
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.actorder = actorder
        self.tile_cols = self.columns if groupsize == -1 else groupsize
        self.scales = torch.zeros(
            (self.rows, (self.columns + self.tile_cols - 1) // self.tile_cols),
            dtype=self.linear_module.weight.dtype,
            device=self.dev,
        )
        self.zeros = torch.zeros_like(self.scales)
        assert not (
            self.actorder and self.groupsize != -1
        ), "The permutation trick does not work for grouped quantization"

    @staticmethod
    def quantize_weight(x, scale, zero, maxq):
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        x_rec = scale * (q - zero)
        return x_rec

    def find_params_weight(self, x):
        dev = x.device

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / self.maxq
        if self.sym:
            zero = torch.full_like(scale, (self.maxq + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)

        if not self.perchannel:
            tmp = shape[0]
            scale = scale.repeat(tmp)
            zero = zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        scale = scale.reshape(shape)
        zero = zero.reshape(shape)
        return scale, zero

    def collect_input_stats(self, _1, inp, _2):
        inp = inp[0].detach()
        self.last_inp = inp
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def quantize(self):
        W = self.linear_module.weight.detach().to(dtype=torch.float, copy=True)

        scale, zero = self.find_params_weight(W)
        self.scales[:] = scale
        self.zeros[:] = zero

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        if self.actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if self.groupsize != -1:
                    if (i1 + i) % self.groupsize == 0:
                        scale, zero = self.find_params_weight(W[:, (i1 + i) : (i1 + i + self.groupsize)])
                        self.scales[:, (i1 + i) // self.groupsize] = scale
                        self.zeros[:, (i1 + i) // self.groupsize] = zero

                q = self.quantize_weight(w.unsqueeze(1), scale, zero, self.maxq)
                q = q.squeeze(1)
                assert q.dim() == 1
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if self.actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        weight = Q.reshape(self.linear_module.weight.shape).to(self.linear_module.weight.data.dtype)
        error = torch.sum(Losses).item()

        from quantize.bnb import ColBlockQuantizedLinear

        q_module = ColBlockQuantizedLinear(
            self.linear_module.in_features,
            self.linear_module.out_features,
            self.linear_module.bias is not None,
            bits=self.bits,
            tile_cols=self.groupsize,
        ).to(self.dev)
        q_module.scales = self.scales
        q_module.zeros = self.zeros
        q_module.pack_weight(weight)
        q_module.bias = self.linear_module.bias
        return q_module, error


def get_sample_data():
    traindata = load_dataset(
        "allenai/c4", "allenai--c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
    )
    # heuristic for the data size?
    txt = "\n".join(traindata[i]["text"] for i in torch.randperm(len(traindata))[:2000].tolist())
    return txt


@torch.no_grad()
def blockwise_quantization(model, sample_inputs, working_device, *, bits=4, groupsize=-1):
    """
    This is the classic post-training quantization of all linear layers.
    We quantize in order, i.e. when observing the inputs, we use the outputs of the previously quantized layers rather
    than doing them all at once.
    """
    print(model)
    print(model.config)

    print("Getting inputs for first block")
    model.transformer.wte.to(working_device)
    sample_inputs = sample_inputs.to(working_device)
    inps = model.transformer.wte(sample_inputs)
    model.transformer.wte.to("cpu")
    torch.cuda.empty_cache()

    rope_cache = model.build_rope_cache(sample_inputs)
    mask_cache = model.build_mask_cache(sample_inputs)

    print("Starting to quantize blocks")
    outs = torch.zeros_like(inps)

    # better than relying on enumeration? originally the code bundled
    # the two mlp fc layers
    # we could automate this with a lot of hooks and another iteration
    submodules_to_process = ["attn.attn", "attn.proj", "mlp.proj"]
    if model.config._mlp_class == "GptNeoxMLP":
        submodules_to_process.append("mlp.fc")
    else:
        submodules_to_process.extend(["mlp.fc_1", "mlp.fc_2"])

    for i, block in enumerate(model.transformer.h):
        block.to(working_device)

        for name in submodules_to_process:
            print(i, name, end=" ")
            t0 = time.perf_counter()
            print("collecting stats", end=" ")
            sys.stdout.flush()
            module = block.get_submodule(name)

            gptq = GPTQQuantizer(module, bits=bits, groupsize=groupsize, actorder=(groupsize == -1))
            handle = module.register_forward_hook(gptq.collect_input_stats)
            for j in range(inps.size(0)):
                outs[j : j + 1], _ = block(
                    inps[j : j + 1], rope=rope_cache, mask=mask_cache, max_seq_length=model.config.block_size
                )

            handle.remove()

            print("quantizing", end=" ")
            sys.stdout.flush()
            q_module, error = gptq.quantize()

            # replace the linear module with the quantized module
            pname, dname = name.rsplit(".", 1)
            setattr(block.get_submodule(pname), dname, q_module)

            # cleanup in an attempt to not run out of memory
            del gptq
            gc.collect()
            torch.cuda.empty_cache()
            t1 = time.perf_counter()
            print(f"time {int(t1 - t0 + 0.5)}s quantization error {error:.1f}")

        for j in range(inps.size(0)):
            outs[j : j + 1], _ = block(
                inps[j : j + 1], rope=rope_cache, mask=mask_cache, max_seq_length=model.config.block_size
            )

        block.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # the outputs are the next block's inputs and we'll reuse the old inputs
        inps, outs = outs, inps

    model.transformer.ln_f.to(working_device)
    for j in range(inps.size(0)):
        outs[j : j + 1] = model.transformer.ln_f(inps[j : j + 1])
    model.transformer.ln_f.to("cpu")
    inps, outs = outs, inps

    model.lm_head.to(working_device)
    gptq = GPTQQuantizer(model.lm_head, bits=bits, groupsize=groupsize, actorder=(groupsize == -1))
    handle = model.lm_head.register_forward_hook(gptq.collect_input_stats)
    for j in range(inps.size(0)):
        model.lm_head(inps[j : j + 1])
    handle.remove()
    q_module, error = gptq.quantize()
    model.lm_head = q_module
    model.lm_head.to("cpu")


def main(
    *,
    checkpoint_dir: Path = Path(f"checkpoints/stabilityai/stablelm-base-alpha-3b"),
    output_path: Optional[Path] = None,
    n_samples: int = 128,
    precision: str = "bf16-true",
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        checkpoint_dir: The checkpoint directory to load.
        output_path: Path to write the quantized model's state dict to.
        n_samples: Number of example inputs to use for statistics (default: 128)
        precision: The precision to use to load the model.
    """
    if output_path is None:
        output_path = checkpoint_dir / "lit_model_gptq.4bit.pth"
    check_valid_checkpoint_dir(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    device = "cuda"
    fabric = Fabric(accelerator="cuda", precision=precision)

    # we avoid loading the entire model on the GPU and do this block by block
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.time()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    with lazy_load(checkpoint_path) as checkpoint:
        model.load_state_dict(checkpoint)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()

    tokenizer = Tokenizer(checkpoint_dir)

    test_string = get_sample_data()
    encoded_text = tokenizer.encode(test_string, eos=True)
    block_size = config.block_size
    encoded_text = encoded_text[: n_samples * block_size].reshape(n_samples, block_size)

    t0 = time.perf_counter()
    blockwise_quantization(model, encoded_text, device, bits=4)
    t = time.perf_counter() - t0

    print(f"\n\nTime for quantization: {t:.02f} sec total", file=sys.stderr)
    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)

    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
