## Config files

The table below lists the performances you can expect from the provided config files. Note that you can achieve lower memory consumption by lowering the micro batch size as needed. In addition, you can lower the rank (`lora_r`) in the LoRA configuration files and disable LoRA for certain layers (for example, setting `lora_projection` and other LoRA layer-specific parameters to `false`).
For more information, see the [Dealing with out-of-memory (OOM) errors](../../tutorials/oom.md) on lowering the memory requirements.
The "Cost" column refers to the on-demand compute cost on [Lightning AI Studios where these benchmarks were executed](https://lightning.ai/lightning-ai/studios/automated-benchmarks-for-litgpt).
All experiments were conducted using bfloat-16 precision on the Alpaca2k dataset. The "Multitask score" refers to [MMLU](https://arxiv.org/abs/2009.03300).

&nbsp;

| Config                            | Model                  | Epochs | Max seq length | Micro batch size | Machine | Training runtime | Cost | Peak memory | Validation loss | Validation perplexity | Multitask score (MMLU) |
| --------------------------------- | ---------------------- | ------ | -------------- | ---------------- | ------- | ---------------- | ---- | ----------- | --------------- | --------------------- | --------------- |
| falcon-7b/lora.yaml               | falcon-7b              | 4      | 512            | 1                | 1xA10G  | 24.84 min        | $0.7 | 16.69 GB    | 0.945           | 2.573                 | 26.2%           |
| falcon-7b/lora.yaml               | falcon-7b              | 4      | 512            | 1                | 4xA10G  | 24.94 min        | $2.0 | 16.69 GB    | 0.945           | 2.573                 | 26.4%           |
| falcon-7b/qlora.yaml              | falcon-7b              | 4      | 512            | 1                | 1xA10G  | 50.85 min        | $1.5 | 9.44 GB     | 0.993           | 2.699                 | 26.3%           |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| gemma-2b/full.yaml                | gemma-2b               | 1      | 512            | 1                | 4xA10G  | 14.06 min        | $1.1 | 17.43 GB    | 1.021           | 2.777                 | 32.4%           |
| gemma-2b/lora.yaml                | gemma-2b               | 2      | 512            | 2                | 1xA10G  | 9.41 min         | $0.3 | 12.62 GB    | 0.981           | 2.666                 | 34.4%           |
| gemma-2b/lora.yaml                | gemma-2b               | 2      | 512            | 2                | 4xA10G  | 9.41 min         | $0.8 | 12.62 GB    | 0.981           | 2.667                 | 34.0%           |
| gemma-2b/qlora.yaml               | gemma-2b               | 2      | 512            | 2                | 1xA10G  | 12.91 min        | $0.4 | 11.58 GB    | 1.085           | 2.959                 | 36.4%           |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| gemma-7b/lora.yaml                | gemma-7b               | 2      | 512            | 1                | 1xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |                 |
| gemma-7b/lora.yaml                | gemma-7b               | 2      | 512            | 1                | 4xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |                 |
| gemma-7b/qlora.yaml               | gemma-7b               | 2      | 512            | 1                | 1xA10G  | 43.58 min        | $1.3 | 17.18 GB    | 0.973           | 2.646                 | 62.45%          |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| gemma2-2b/lora.yaml               | gemma-2b               | 2      | 512            | 2                | 1xA10G  | 11.96 min        | $0.4 | 14.31 GB    | 0.951           | 2.589                 | 23.84%          |
| gemma2b/qlora.yaml                | gemma-2b               | 2      | 512            | 2                | 1xA10G  | 16.06 min        | $0.5 | 13.52 GB    | 0.983           | 2.673                 | 24.12%          |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| gemma2-9b/lora.yaml               | gemma-2-9b             | 2      | 512            | 1                | 1xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |                 |
| gemma2-9b/lora.yaml               | gemma-2-9b             | 2      | 512            | 1                | 4xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |                 |
| gemma2-9b/qlora.yaml              | gemma-2-9b             | 2      | 512            | 1                | 1xA10G  | 50.01 min        | $4.0 | 20.92 GB    | 0.852           | 2.345                 | 24.2%           |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| llama-2-7b/full.yaml              | llama-2-7b             | 1      | 512            | 4                | 4xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |                 |
| llama-2-7b/lora.yaml              | llama-2-7b             | 4      | 512            | 2                | 1xA10G  | 32.82 min        | $1.0 | 19.77 GB    | 0.802           | 2.230                 | 40.3%           |
| llama-2-7b/lora.yaml              | llama-2-7b             | 4      | 512            | 2                | 4xA10G  | 32.83 min        | $2.6 | 19.77 GB    | 0.802           | 2.229                 | 40.2%           |
| llama-2-7b/qlora.yaml             | llama-2-7b             | 4      | 512            | 2                | 1xA10G  | 45.67 min        | $1.4 | 13.68 GB    | 0.814           | 2.258                 | 38.6%           |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| llama-3-8b/full.yaml              | llama-3-8b             | 1      | 512            | 4                | 4xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |                 |
| llama-3-8b/lora.yaml              | llama-3-8b             | 2      | 512            | 1                | 1xA10G  | 14.79 min        | $0.4 | 19.73 GB    | 0.888           | 2.431                 | 62.4%           |
| llama-3-8b/lora.yaml              | llama-3-8b             | 2      | 512            | 1                | 4xA10G  | 14.88 min        | $1.2 | 19.73 GB    | 0.889           | 2.432                 | 62.5%           |
| llama-3-8b/qlora.yaml             | llama-3-8b             | 2      | 512            | 2                | 1xA10G  | 22.24 min        | $0.7 | 17.41 GB    | 0.939           | 2.558                 | 62.2%           |
|                                   |                        |        |                |                  |         |                  |      |            |                 |                        |                 |
| llama-3.1-8b/full.yaml            | llama-3.1-8b           | 1      | 512            | 4                | 1xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   | OOM             |
| llama-3.1-8b/lora.yaml            | llama-3.1-8b           | 2      | 512            | 1                | 1xA10G  | 13.36 min        | $1.1 | 19.73 GB    | 0.878           | 2.406                 | xx.xx           |
| llama-3.1-8b/qlora.yaml           | llama-3.1-8b           | 2      | 512            | 2                | 1xA10G  | 21.81 min        | $0.7 | 17.41 GB    | 0.928           | 2.529                 | xx.xx           |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| llama-3.2-1b/full.yaml            | llama-3.2-1b           | 1      | 512            | 4                | 1xA10G  |  2.01 min        | $0.1 |  8.70 GB    | 1.442           | 4.229                 | 38.21%          |
| llama-3.2-1b/lora.yaml            | llama-3.2-1b           | 2      | 512            | 1                | 1xA10G  |  4.17 min        | $0.4 |  4.49 GB    | 1.114           | 3.046                 | 36.87%          |
| llama-3.2-1b/qlora.yaml           | llama-3.2-1b           | 2      | 512            | 2                | 1xA10G  |  6.20 min        | $0.6 |  5.53 GB    | 1.201           | 3.322                 | 36.49%          |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| llama-3.2-3b/full.yaml            | llama-3.2-3b           | 1      | 512            | 4                | 1xA10G  |  4.71 min        | $0.4 | 16.51 GB    | 1.255           | 3.509                 | 54.69%          |
| llama-3.2-3b/lora.yaml            | llama-3.2-3b           | 2      | 512            | 1                | 1xA10G  |  8.31 min        | $0.8 |  9.67 GB    | 0.973           | 2.647                 | 54.77%          |
| llama-3.2-3b/qlora.yaml           | llama-3.2-3b           | 2      | 512            | 2                | 1xA10G  | 14.89 min        | $1.4 | 10.30 GB    | 1.031           | 2.804                 | 55.08%          |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| mistral-7b-v0.2/lora.yaml         | mistral-7b-v0.2        | 4      | 512            | 2                | 1xA10G  | 31.00 min        | $0.9 | 20.66 GB    | 0.801           | 2.228                 | 55.7%           |
| mistral-7b-v0.2/lora.yaml         | mistral-7b-v0.2        | 4      | 512            | 2                | 4xA10G  | 31.00 min        | $2.5 | 20.66 GB    | 0.802           | 2.229                 | 55.5%           |
| mistral-7b-v0.2/qlora.yaml        | mistral-7b-v0.2        | 4      | 512            | 2                | 1xA10G  | 44.75 min        | $1.3 | 14.29 GB    | 0.813           | 2.255                 | 56.5%           |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| mistral-7b/lora.yaml              | mistral-7b             | 4      | 512            | 2                | 1xA10G  | 31.01 min        | $0.9 | 20.66 GB    | 0.794           | 2.211                 | 57.9%           |
| mistral-7b/lora.yaml              | mistral-7b             | 4      | 512            | 2                | 4xA10G  | 31.03 min        | $2.5 | 20.66 GB    | 0.796           | 2.218                 | 57.9%           |
| mistral-7b/qlora.yaml             | mistral-7b             | 4      | 512            | 2                | 1xA10G  | 44.75 min        | $1.3 | 14.29 GB    | 0.803           | 2.231                 | 57.9%           |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| phi-2/full.yaml                   | phi-2                  | 1      | 512            | 4                | 4xA10G  | 11.87 min        | $1.0 | 14.44 GB    | 1.305           | 3.688                 | 38.4%           |
| phi-2/lora.yaml                   | phi-2                  | 1      | 512            | 4                | 1xA10G  | 3.78 min         | $0.1 | 13.98 GB    | 0.819           | 2.269                 | 53.0%           |
| phi-2/lora.yaml                   | phi-2                  | 1      | 512            | 4                | 4xA10G  | 3.78 min         | $0.3 | 13.98 GB    | 0.820           | 2.271                 | 52.4%           |
| phi-2/qlora.yaml                  | phi-2                  | 1      | 512            | 4                | 1xA10G  | 4.51 min         | $0.1 | 14.27 GB    | 0.837           | 2.310                 | 52.3%           |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| phi-3/full.yaml                   | Phi-3-mini-4k-instruct | 1      | 512            | 4                | 1xA10G  | 6.93 min         | $0.2 | 17.01 GB    | 0.714           | 2.043                 | 69.81%          |
| phi-3/lora.yaml                   | Phi-3-mini-4k-instruct | 1      | 512            | 4                | 1xA10G  | 6.46 min         | $0.2 | 19.75 GB    | 0.707           | 2.028                 | 69.70%          |
| phi-3/qlora.yaml                  | Phi-3-mini-4k-instruct | 1      | 512            | 4                | 1xA10G  | 7.47 min         | $0.2 | 19.13 GB    | 0.729           | 2.074                 | 68.96%          |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| stablelm-base-alpha-3b/full.yaml  | stablelm-base-alpha-3b | 1      | 512            | 1                | 4xA10G  | 70.13 min        | $5.6 | 21.23 GB    | 1.513           | 4.540                 | 23.2%           |
| stablelm-base-alpha-3b/lora.yaml  | stablelm-base-alpha-3b | 4      | 512            | 1                | 1xA10G  | 13.07 min        | $0.4 | 8.58 GB     | 1.361           | 3.900                 | 25.9%           |
| stablelm-base-alpha-3b/lora.yaml  | stablelm-base-alpha-3b | 4      | 512            | 1                | 4xA10G  | 13.16 min        | $1.1 | 8.58 GB     | 1.362           | 3.906                 | 25.9%           |
| stablelm-base-alpha-3b/qlora.yaml | stablelm-base-alpha-3b | 4      | 512            | 1                | 1xA10G  | 25.86 min        | $0.8 | 5.24 GB     | 1.388           | 4.009                 | 26.1%           |
|                                   |                        |        |                |                  |         |                  |      |             |                 |                       |                 |
| tiny-llama/full.yaml              | tiny-llama             | 1      | 512            | 4                | 1xA10G  | 2.58 min         | $0.1 | 14.10 GB    | 1.088           | 2.968                 | 24.6%           |
| tiny-llama/full.yaml              | tiny-llama             | 1      | 512            | 4                | 4xA10G  | 2.57 min         | $0.2 | 14.10 GB    | 1.088           | 2.968                 | 24.5%           |
| tiny-llama/lora.yaml              | tiny-llama             | 3      | 512            | 8                | 1xA10G  | 8.09 min         | $0.2 | 13.50 GB    | 1.039           | 2.826                 | 25.5%           |
| tiny-llama/qlora.yaml             | tiny-llama             | 3      | 512            | 8                | 1xA10G  | 8.70 min         | $0.3 | 16.24 GB    | 1.056           | 2.874                 | 25.3%           |

*OOM = Out of memory


&nbsp;
## Extending the context length

If you require a longer sequence length than the one used in a given config file, you can either edit the `max_seq_length` in the config file or pass an additional argument when running the finetuning command, for example, `--max_seq_length 4096` to override the sequence length provided in the config file.

&nbsp;
## Training on GPUs without bfloat16 support

If you are training on GPUs without bfloat-16 support, you need to change the `precision` option to `16-true` (16-bit floating point precision) or `16-mixed` (16/32-bit mixed precision) training:

```bash
litgpt finetune lora \
  --config config_hub/finetune/phi-2/lora.yaml \
  --precision 16-true
```
or

```bash
litgpt finetune lora \
  --config config_hub/finetune/phi-2/lora.yaml \
  --precision 16-mixed
```

Note that `16-true` is more compute and memory-efficient, but it can sometimes lead to training convergence issues. In this case, it's recommended to use `16-mixed`.

&nbsp;
## Multi-GPU experiments

All runs are single-GPU experiments, use `--devices 4` to utilize more than one GPU:


```bash
litgpt finetune lora \
  --config config_hub/finetune/phi-2/lora.yaml \
  --devices 4
```
