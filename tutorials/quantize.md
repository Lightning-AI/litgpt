# Quantize the model

This document provides different strategies for quantizing the various models available in Lit-GPT to reduce GPU memory usage, which is useful for running larger models on certain GPU hardware.

**All the examples below were run on an A100 40GB GPU.**

> [!NOTE]\:
> Quantization is only supported with inference (generate and chat scripts).


## Baseline

It's useful to start with a baseline to have a reference point for memory savings via the various quantization methods.

```bash
python generate/base.py --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision 32-true --max_new_tokens 256
...
Time for inference 1: 10.69 sec total, 23.95 tokens/sec.
Memory used: 28.95 GB
```

First, using a lower precision compared to 32-bit float can result in two times reduced memory consumption. You can either try setting `--precision 16-true` for regular 16-bit precision or  `--precision bf16-true` if your GPU supports brain-float 16-bit precision. ([This brief video](https://lightning.ai/courses/deep-learning-fundamentals/9.0-overview-techniques-for-speeding-up-model-training/unit-9.1-accelerated-model-training-via-mixed-precision-training/) explains the difference between regular 16-bit and bf16-bit precision.)

In short, when `--precision bf16-true` or `--precision 16-true` is used, the model weights will automatically be converted and consume less memory.
However, this might not be enough for large models or when using GPUs with limited memory.

```bash
python generate/base.py --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 9.76 sec total, 26.23 tokens/sec.
Memory used: 14.51 GB
````

To reduce the memory requirements further, Lit-GPT supports several quantization techniques, which are shown below.

> [!NOTE]\:
> Most quantization examples below also use the `--precision bf16-true` setting explained above. If your GPU does not support the bfloat-format, you can change it to `--precision 16-true`.

## `bnb.nf4`

Enabled with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

> [!NOTE]\: `bitsandbytes` only supports `CUDA` devices and the `Linux` operating system.
> Windows users should use [WSL2](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).

Uses the normalized float 4 (nf4) data type. This is recommended over "fp4" based on the paper's experimental results and theoretical analysis.

```bash
python generate/base.py --quantize bnb.nf4 --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 8.92 sec total, 28.69 tokens/sec
Memory used: 5.72 GB
```

## `bnb.nf4-dq`

Enabled with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

"dq" stands for "Double Quantization" which reduces the average memory footprint by quantizing the quantization constants.
In average, this amounts to about 0.37 bits per parameter (approximately 3 GB for a 65B model).

```bash
python generate/base.py --quantize bnb.nf4-dq --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 12.06 sec total, 21.23 tokens/sec
Memory used: 5.37 GB
```

## `bnb.fp4`

Enabled with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

Uses pure FP4 quantization.

```bash
python generate/base.py --quantize bnb.fp4 --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 9.20 sec total, 27.83 tokens/sec
Memory used: 5.72 GB
```

## `bnb.fp4-dq`

Enabled with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

"dq" stands for "Double Quantization" which reduces the average memory footprint by quantizing the quantization constants.
In average, this amounts to about 0.37 bits per parameter (approximately 3 GB for a 65B model).

```bash
python generate/base.py --quantize bnb.fp4-dq --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 12.12 sec total, 21.13 tokens/sec
Memory used: 5.37 GB
```

## `bnb.int8`

Enabled with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2110.02861) to learn more about how it works.

```bash
python generate/base.py --quantize bnb.int8 --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 24.17 sec total, 10.59 tokens/sec
Memory used: 8.71 GB
```

## `gptq.int4`

Check out the [paper](https://arxiv.org/abs/2210.17323) to learn more about how it works.

This technique needs a conversion of the weights first:

```bash
python quantize/gptq.py --precision bf16-true --checkpoint_dir checkpoints/tiiuae/falcon-7b
...
Time for quantization: 850.25 sec total
Memory used: 23.68 GB
```

It is important to note that this conversion step required a considerable amount of memory (higher than regular inference) and may take a long time, depending on the size of the model.

generation then works as usual with `--quantize gptq.int4` which will load the newly quantized checkpoint file:

```bash
python generate/base.py --quantize gptq.int4 --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision 32-true --max_new_tokens 256
...
Time for inference 1: 39.50 sec total, 6.48 tokens/sec
Memory used: 5.05 GB
```
