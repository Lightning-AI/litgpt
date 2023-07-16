# Quantize the model

When `--precision bf16-true` or `--precision 16-true` is used, the model weights will automatically be converted and consume less memory.
However, this might not be enough for large models or when using GPUs with limited memory.

> **Note**: 
> Quantization is only supported with inference (generate and chat scripts).

### Baseline

All the examples below were run on an A100 40GB GPU.

```bash
python generate/base.py --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 9.76 sec total, 26.23 tokens/sec.
Memory used: 14.51 GB
```

To reduce the memory requirements further, Lit-GPT supports several quantization techniques:

## `bnb.nf4`

Enabled with [bitsandbyes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

Uses the normalized float 4 (nf4) data type. This is recommended over "fp4" based on the paper's experimental results and theoretical analysis. 

```bash
python generate/base.py --quantize bnb.nf4 --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 8.92 sec total, 28.69 tokens/sec
Memory used: 5.72 GB
```

## `bnb.nf4-dq`

Enabled with [bitsandbyes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

"dq" stands for "Double Quantization" which reduces the average memory footprint by quantizing the quantization constants.
In average, this amounts to about 0.37 bits per parameter (approximately 3 GB for a 65B model).

```bash
python generate/base.py --quantize bnb.nf4-dq --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 12.06 sec total, 21.23 tokens/sec
Memory used: 5.37 GB
```

## `bnb.fp4`

Enabled with [bitsandbyes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

Uses pure FP4 quantization.

```bash
python generate/base.py --quantize bnb.fp4 --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 9.20 sec total, 27.83 tokens/sec
Memory used: 5.72 GB
```

## `bnb.fp4-dq`

Enabled with [bitsandbyes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2305.14314v1) to learn more about how it works.

"dq" stands for "Double Quantization" which reduces the average memory footprint by quantizing the quantization constants.
In average, this amounts to about 0.37 bits per parameter (approximately 3 GB for a 65B model).

```bash
python generate/base.py --quantize bnb.fp4-dq --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --max_new_tokens 256
...
Time for inference 1: 12.12 sec total, 21.13 tokens/sec
Memory used: 5.37 GB
```

## `bnb.int8`

Enabled with [bitsandbyes](https://github.com/TimDettmers/bitsandbytes). Check out the [paper](https://arxiv.org/abs/2110.02861) to learn more about how it works.

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
