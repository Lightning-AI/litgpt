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

import torch
from datasets import load_dataset
from lightning import Fabric

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Tokenizer, Config
from lit_gpt.utils import check_valid_checkpoint_dir, lazy_load

import triton
import triton.language as tl


# This is adapted from the OpenAI Triton matmul example.
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_kernel_4bit_weight(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    bscales_ptr,
    bzeros_ptr,
    # bdequant,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.T.
    A has shape (M, K), B has shape (N, K) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_n] pointers
    # see above `Pointer Arithmetics` section for details
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_mask = offs_am[:, None] < M
    b_mask = offs_bn[None, :] < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 2) * stride_bk + offs_bn[None, :] * stride_bn)

    bscales_ptrs = bscales_ptr + offs_bn[None, :]
    bzeros_ptrs = bzeros_ptr + offs_bn[None, :]

    scale = tl.load(bscales_ptrs)
    zero = tl.load(bzeros_ptrs)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # wasteful as it is to load everything twice, my attempts at avoiding it lead to slower code
        b12 = tl.load(b_ptrs, mask=b_mask)
        # Note that for simplicity, we don't apply a mask in K here.
        a = tl.load(a_ptrs, mask=a_mask).to(tl.float32)
        b = (((b12.to(tl.uint8) >> ((offs_k[:, None] % 2) * 4)) & 0xF).to(tl.float32) - zero) * scale
        accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
    c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def qlinear_4bit_weight(inp, weight, scales, zeros):
    weight = weight.t().contiguous()
    c_shape = inp.shape[:-1] + weight.shape[-1:]
    inp = inp.reshape(-1, inp.shape[-1]).contiguous()
    # we pad the input to amortize triton compilation cost better
    PAD_TO = 256
    if inp.shape[0] % PAD_TO != 0:
        c_crop = inp.shape[0]
        new_inp_shape0 = inp.shape[0] + PAD_TO - inp.shape[0] % PAD_TO
        inp2 = inp.new_empty((new_inp_shape0, inp.shape[1]))
        inp2[: inp.shape[0]] = inp
        inp2[inp.shape[0] :].zero_()
        inp = inp2
    else:
        c_crop = None

    assert inp.shape[1] == weight.shape[0] * 2, "incompatible dimensions"

    assert scales.shape == (weight.shape[1], 1)
    assert zeros.shape == (weight.shape[1], 1)
    scales = scales.contiguous()
    zeros = zeros.contiguous()
    K, N = weight.shape
    M, K = inp.shape
    assert K % 32 == 0, "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((M, N), device=inp.device, dtype=inp.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    linear_kernel_4bit_weight[grid](
        inp,
        weight,
        c,
        scales,
        zeros,
        M,
        N,
        K,
        inp.stride(0),
        inp.stride(1),
        weight.stride(0),
        weight.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c[:c_crop].reshape(c_shape)


# for correctness but with terrible perf
class ColBlockQuantizedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias: bool, *, bits, tile_cols):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_cols = tile_cols if tile_cols != -1 else self.in_features
        self.bits = bits
        self.entries_per_byte = 8 // bits
        assert self.entries_per_byte > 0
        assert self.entries_per_byte * self.bits == 8
        assert in_features % self.entries_per_byte == 0
        self.register_buffer(
            "quant_weight",
            torch.empty((self.out_features, self.in_features // self.entries_per_byte), dtype=torch.uint8)
            .t()
            .contiguous()
            .t(),
        )
        self.register_buffer(
            "scales", torch.empty((self.out_features, (self.in_features + self.tile_cols - 1) // self.tile_cols))
        )
        self.register_buffer("zeros", torch.empty_like(self.scales))
        assert isinstance(bias, bool)
        if bias:
            self.register_buffer("bias", torch.empty((self.out_features,)))
        else:
            self.register_buffer("bias", None)

    def pack_weight(self, weight):
        weight = weight.to(device=self.quant_weight.device, copy=True)
        for j in range(self.scales.size(1)):
            weight[:, j * self.tile_cols : (j + 1) * self.tile_cols] /= self.scales[:, j : j + 1]
            weight[:, j * self.tile_cols : (j + 1) * self.tile_cols] += self.zeros[:, j : j + 1]
        weight = weight.clamp_(min=0, max=2**self.bits - 1).to(dtype=torch.uint8)
        self.quant_weight.zero_()
        for nr in range(self.entries_per_byte):
            self.quant_weight += weight[:, nr :: self.entries_per_byte] << (nr * self.bits)

    def get_weight(self, dtype=torch.float):
        weight = torch.empty((self.out_features, self.in_features), device=self.quant_weight.device, dtype=dtype)
        mask = (1 << self.bits) - 1
        for nr in range(self.entries_per_byte):
            weight[:, nr :: self.entries_per_byte] = ((self.quant_weight >> (nr * self.bits)) & mask).float()
        self.quant_weight.to(dtype)
        for j in range(self.scales.size(1)):
            weight[:, j * self.tile_cols : (j + 1) * self.tile_cols] -= self.zeros[:, j : j + 1]
            weight[:, j * self.tile_cols : (j + 1) * self.tile_cols] *= self.scales[:, j : j + 1]
        return weight

    def forward(self, inp):
        if (
            triton is not None
            and self.bits == 4
            and self.quant_weight.device.type == "cuda"
            and self.zeros.shape[1] == 1
            and self.quant_weight.shape[1] % 32 == 0
        ):
            return qlinear_4bit_weight(inp, self.quant_weight, self.scales, self.zeros)
        weight = self.get_weight(dtype=inp.dtype)
        return torch.nn.functional.linear(inp, weight, self.bias)


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
        return scale * (q - zero)

    def find_params_weight(self, x):
        dev = x.device

        shape = x.shape
        x = x.flatten(1) if self.perchannel else x.flatten().unsqueeze(0)

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
        zero = torch.full_like(scale, (self.maxq + 1) / 2) if self.sym else torch.round(-xmin / scale)

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

                if self.groupsize != -1 and (i1 + i) % self.groupsize == 0:
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
    return "\n".join(traindata[i]["text"] for i in torch.randperm(len(traindata))[:2000].tolist())


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
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
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
