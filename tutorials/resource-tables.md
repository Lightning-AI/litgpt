# Resource Tables

- Last updated: 10/20/2023
- Lit-GPT version: commit 8641822
- Hardware: NVIDIA A100-SXM4-40GB
- OS: Ubuntu 22.04.3 LTS (x86_64)
- Nvidia driver version: 525.125.06
- Relevant libraries
  - PyTorch 2.1.0+cu121
  - Bitsandbytes 0.41.1

This document provides an overview and examples of hardware requirements when running models in Lit-GPT.

For additional tips on lowering the GPU memory footprint, please also see the [Dealing with out-of-memory (OOM) errors](oom.md) document.

All experiments were run using 16-bit brain floating point precision (`--precision bf16-true`). If your GPU does not support brain floating point precision, you can use regular 16-bit floating point precision (`--precision 16-true`).

All experiments were conducted using the Alpaca dataset with its default length. Note that due to different tokenizers being used by the different models, the number of tokens in the longest training example differs based on the model:

- phi1.5: 1044 tokens
- StableLM Alpha: 1034 tokens
- Llama 2: 1304 tokens
- Falcon 1079 tokens

Note that the number of tokens in the training set does not affect the supported context width (block size) of the models, which is as follows:

- phi1.5: 2048 tokens
- StableLM 3B Alpha: 4096 tokens
- Llama 2: 4048 tokens
- Falcon: 2048 tokens
- CodeLlama 13B: 16384 tokens

&nbsp;

## Finetuning with LoRA on 1 GPU

The following experiments were conducted on 1xA100 with a minibatch size of 128 using the `finetune/lora.py` script.

| Size  | Model          | Quantization | Microbatch size | Trainable parameters | Max GPU RAM | Time 1k iterations |
|-------|----------------|--------------|-----------------|----------------------|-------------|--------------------|
| 1.3 B | phi-1.5        | None         | 1               | 1,572,864            | 4.82 GB     | 1.62 min           |
| 1.3 B | phi-1.5        | bnb.nf4      | 1               | 1,572,864            | 3.78 GB     | 1.77 min           |
| 1.3 B | phi-1.5        | bnb.nf4-dq   | 1               | 1,572,864            | 3.72 GB     | 1.87 min           |
| 1.3 B | phi-1.5        | None         | 2               | 1,572,864            | 6.76 GB     | 1.65 min           |
| 1.3 B | phi-1.5        | None         | 4               | 1,572,864            | 10.68 GB    | 1.70 min           |
|       |                |              |                 |                      |             |                    |
| 3 B   | StableLM Alpha | None         | 1               | 2,097,152            | 9.69 GB     | 1.24 min           |
| 3 B   | StableLM Alpha | bnb.nf4      | 1               | 2,097,152            | 6.35 GB     | 1.82 min           |
| 3 B   | StableLM Alpha | bnb.nf4-dq   | 1               | 2,097,152            | 6.19 GB     | 1.87 min           |
| 3 B   | StableLM Alpha | None         | 2               | 2,097,152            | 12.10 GB    | 1.33 min           |
| 3 B   | StableLM Alpha | None         | 4               | 2,097,152            | 16.92 GB    | 1.50 min           |
|       |                |              |                 |                      |             |                    |
| 7 B   | Llama 2        | None         | 1               | 4,194,304            | 21.30 GB    | 2.36 min           |
| 7 B   | Llama 2        | bnb.nf4      | 1               | 4,194,304            | 14.14 GB    | 3.68 min           |
| 7 B   | Llama 2        | bnb.nf4-dq   | 1               | 4,194,304            | 13.84 GB    | 3.83 min           |
| 7 B   | Llama 2        | None         | 2               | 4,194,304            | 29.07 GB    | 2.52 min           |
| 7 B   | Llama 2        | None         | 4               | 4,194,304            | OOM         | -                  |
|       |                |              |                 |                      |             |                    |
| 13 B  | Llama 2        | None         | 1               | 6,553,600            | 38.12 GB    | 3.19 min           |
| 13 B  | Llama 2        | bnb.nf4      | 1               | 6,553,600            | 23.14 GB    | 6.38 min           |
| 13 B  | Llama 2        | bnb.nf4-dq   | 1               | 6,553,600            | 22.55 GB    | 6.55 min           |
| 13 B  | Llama 2        | None         | 2               | 6,553,600            | OOM         | -                  |
| 13 B  | Llama 2        | None         | 4               | 6,553,600            | OOM         | -                  |
|       |                |              |                 |                      |             |                    |
| 40 B  | Falcon         | None         | 1               | 12,042,240           | OOM         | -                  |
| 40 B  | Falcon         | bnb.nf4      | 1               | 12,042,240           | OOM         | -                  |
| 40 B  | Falcon         | bnb.nf4-dq   | 1               | 12,042,240           | OOM         | -                  |

&nbsp;

## Finetuning with Adapter on 1 GPU

The following experiments were conducted on 1xA100 with a minibatch size of 128 using the `finetune/adapter.py` script.

| Size | Model          | Quantization | Microbatch size | Trainable parameters | Max GPU RAM | Time 1k iterations |
|------|----------------|--------------|-----------------|----------------------|-------------|--------------------|
| 3 B  | StableLM Alpha | None         | 1               | 573,888              | 9.10 GB     | 0.74 min           |
| 3 B  | StableLM Alpha | bnb.nf4      | 1               | 573,888              | 5.65 GB     | 1.38 min           |
| 3 B  | StableLM Alpha | bnb.nf4-dq   | 1               | 573,888              | 5.48 GB     | 1.46 min           |
|      |                |              |                 |                      |             |                    |
| 7 B  | Llama 2        | None         | 1               | 1,229,760            | 19.98 GB    | 1.50 min           |
| 7 B  | Llama 2        | bnb.nf4      | 1               | 1,229,760            | 12.68 GB    | 2.93 min           |
| 7 B  | Llama 2        | bnb.nf4-dq   | 1               | 1,229,760            | 12.38 GB    | 3.00 min           |

The same config, but using the `finetune/adapter_v2.py` script.

| Size | Model          | Quantization | Microbatch size | Trainable parameters | Max GPU RAM | Time 1k iterations |
|------|----------------|--------------|-----------------|----------------------|-------------|--------------------|
| 3 B  | StableLM Alpha | None         | 1               | 2,125,248            | 10.71 GB    | 0.87 min           |
| 3 B  | StableLM Alpha | bnb.nf4      | 1               | 2,125,248            | 7.41 GB     | 1.59 min           |
| 3 B  | StableLM Alpha | bnb.nf4-dq   | 1               | 2,125,248            | 7.25 GB     | 1.62 min           |
|      |                |              |                 |                      |             |                    |
| 7 B  | Llama 2        | None         | 1               | 4,279,744            | 25.51 GB    | 1.81 min           |
| 7 B  | Llama 2        | bnb.nf4      | 1               | 4,279,744            | 18.30 GB    | 3.23 min           |
| 7 B  | Llama 2        | bnb.nf4-dq   | 1               | 4,279,744            | 17.98 GB    | 3.32 min           |

&nbsp;

## Finetuning with LoRA on Multiple GPUs

The following experiments were conducted on multiple A100 GPUs with a minibatch size of 128 using the `finetune/lora.py` script.

| Size  | Model          | Quantization | Microbatch size | Trainable parameters | GPU      | Max GPU RAM | Time 1k iterations |
|-------|----------------|--------------|-----------------|----------------------|----------|-------------|--------------------|
| 1.3 B | phi-1.5        | None         | 1               | 1,572,864            | 2 x A100 | 4.86 GB     | 3.81 min           |
| 1.3 B | phi-1.5        | bnb.nf4      | 1               | 1,572,864            | 2 x A100 | N/A         | -                  |
| 1.3 B | phi-1.5        | bnb.nf4-dq   | 1               | 1,572,864            | 2 x A100 | N/A         | -                  |
| 1.3 B | phi-1.5        | None         | 2               | 1,572,864            | 2 x A100 | 5.05 GB     | 3.63 min           |
| 1.3 B | phi-1.5        | None         | 4               | 1,572,864            | 2 x A100 | 5.88 GB     | 3.64 min           |
|       |                |              |                 |                      |          |             |                    |
| 3 B   | StableLM Alpha | None         | 1               | 2,097,152            | 2 x A100 | 12.75 GB    | 2.92 min           |
| 3 B   | StableLM Alpha | None         | 2               | 2,097,152            | 2 x A100 | 12.94 GB    | 3.06 min           |
| 3 B   | StableLM Alpha | None         | 4               | 2,097,152            | 2 x A100 | 13.45 GB    | 3.86 min           |
|       |                |              |                 |                      |          |             | -                  |
| 7 B   | Llama 2        | None         | 1               | 4,194,304            | 2 x A100 | 22.18 GB    | 5.93 min           |
| 7 B   | Llama 2        | None         | 2               | 4,194,304            | 2 x A100 | 22.47 GB    | 6.48 min           |
| 7 B   | Llama 2        | None         | 4               | 4,194,304            | 2 x A100 | 23.39 GB    | 8.66 min           |
|       |                |              |                 |                      |          |             |                    |
| 13 B  | Llama 2        | None         | 1               | 6,553,600            | 2 x A100 | OOM         | -                  |
| 13 B  | Llama 2        | bnb.nf4      | 1               | 6,553,600            | 2 x A100 | N/A         | -                  |
| 13 B  | Llama 2        | bnb.nf4-dq   | 1               | 6,553,600            | 2 x A100 | N/A         | -                  |
|       |                |              |                 |                      |          |             |                    |
| 13 B  | Llama 2        | None         | 1               | 6,553,600            | 4 x A100 | 35.57 GB    | 10.25 min          |
| 40 B  | Falcon         | None         | 1               | 12,042,240           | 4 x A100 | OOM         | -                  |

&nbsp;

## Single-GPU Inference

| Size  | Model          | Quantization | GPU      | Max GPU RAM                               | Token/sec |
|-------|----------------|--------------|----------|-------------------------------------------|-----------|
| 1.3 B | phi-1.5        | None         | 1 x A100 | 2.86 GB                                   | 42.56     |
| 1.3 B | phi-1.5        | bnb.nf4      | 1 x A100 | 1.39 GB                                   | 22.89     |
| 1.3 B | phi-1.5        | bnb.nf4-dq   | 1 x A100 | 1.33 GB                                   | 22.75     |
|       |                |              |          |                                           |           |
| 3 B   | StableLM Alpha | None         | 1 x A100 | 7.30 GB                                   | 49.01     |
| 3 B   | StableLM Alpha | bnb.nf4      | 1 x A100 | 3.20 GB                                   | 29.04     |
| 3 B   | StableLM Alpha | bnb.nf4-dq   | 1 x A100 | 3.04 GB                                   | 27.15     |
|       |                |              |          |                                           |           |
| 7 B   | Llama 2        | None         | 1 x A100 | 13.52 GB                                  | 30.97     |
| 7 B   | Llama 2        | bnb.nf4      | 1 x A100 | 4.57 GB                                   | 19.98     |
| 7 B   | Llama 2        | bnb.nf4-dq   | 1 x A100 | 4.26 GB                                   | 17.3      |
|       |                |              |          |                                           |           |
| 13 B  | Llama 2        | None         | 1 x A100 | 26.21 GB                                  | 24.82     |
| 13 B  | Llama 2        | bnb.nf4      | 1 x A100 | 8.32 GB                                   | 16.73     |
| 13 B  | Llama 2        | bnb.nf4-dq   | 1 x A100 | 7.72 GB                                   | 14.43     |
|       |                |              |          |                                           |           |
| 34 B  | CodeLlama      | None         | 1 x A100 | OOM                                       | -         |
| 34 B  | CodeLlama      | bnb.nf4      | 1 x A100 | 20.52 GB                                  | 14.32     |
| 34 B  | CodeLlama      | bnb.nf4-dq   | 1 x A100 | 18.95 GB                                  | 12.37     |
|       |                |              |          |                                           |           |
| 40 B  | Falcon         | None         | 1 x A100 | OOM                                       | -         |
| 40 B  | Falcon         | bnb.nf4      | 1 x A100 | 26.55 GB                                  | 13.25     |
| 40 B  | Falcon         | bnb.nf4-dq   | 1 x A100 | 24.63 GB                                  | 11.64     |
|       |                |              |          |                                           |           |
| 70 B  | Llama 2        | None         | 1 x A100 | OOM                                       | -         |
| 70 B  | Llama 2        | bnb.nf4      | 1 x A100 | CUDA error: CUBLAS_STATUS_NOT_INITIALIZED | -         |
| 70 B  | Llama 2        | bnb.nf4-dq   | 1 x A100 | 37.21 GB                                  | 7.97      |
