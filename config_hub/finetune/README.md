## Config files

The table below lists the performances you can expect from the provided config files. Note that you can achieve lower memory consumption by lowering the micro batch size as needed. In addition, you can lower the rank (`lora_r`) in the LoRA configuration files and disable LoRA for certain layers (for example, setting `lora_projection` and other LoRA layer-specific parameters to `false`).
For more information, see the [Dealing with out-of-memory (OOM) errors](../../tutorials/oom.md) on lowering the memory requirements.
The "Cost" column refers to the on-demand compute cost on [Lightning AI](https://lightning.ai) where these benchmarks were executed.

&nbsp;

| Config                            | Model                  | Dataset  | Epochs | Max seq length | Micro batch size | Precision | Machine | Training runtime | Cost | Peak memory | Validation loss | Validation perplexity |
| --------------------------------- | ---------------------- | -------- | ------ | -------------- | ---------------- | --------- | ------- | ---------------- | ---- | ----------- | --------------- | --------------------- |
| falcon-7b/lora.yaml               | falcon-7b              | Alpaca2k | 4      | 512            | 1                | bf16-true | 1xA10G  | 24.84 min        | $0.7 | 16.69 GB    | 0.945           | 2.573                 |
| falcon-7b/lora.yaml               | falcon-7b              | Alpaca2k | 4      | 512            | 1                | bf16-true | 4xA10G  | 24.94 min        | $2.0 | 16.69 GB    | 0.945           | 2.573                 |
| falcon-7b/qlora.yaml              | falcon-7b              | Alpaca2k | 4      | 512            | 1                | bf16-true | 1xA10G  | 50.85 min        | $1.5 | 9.44 GB     | 0.993           | 2.699                 |
| falcon-7b/qlora.yaml              | falcon-7b              | Alpaca2k | 4      | 512            | 1                | bf16-true | 4xA10G  | 50.88 min        | $4.1 | 9.44 GB     | 0.993           | 2.699                 |
|                                   |                        |          |        |                |                  |           |         |                  |      |             |                 |                       |
| gemma-2b/full.yaml                | gemma-2b               | Alpaca2k | 1      | 512            | 1                | bf16-true | 4xA10G  | 14.06 min        | $1.1 | 17.43 GB    | 1.021           | 2.777                 |
| gemma-2b/lora.yaml                | gemma-2b               | Alpaca2k | 2      | 512            | 2                | bf16-true | 1xA10G  | 9.41 min         | $0.3 | 12.62 GB    | 0.981           | 2.666                 |
| gemma-2b/lora.yaml                | gemma-2b               | Alpaca2k | 2      | 512            | 2                | bf16-true | 4xA10G  | 9.41 min         | $0.8 | 12.62 GB    | 0.981           | 2.667                 |
| gemma-2b/qlora.yaml               | gemma-2b               | Alpaca2k | 2      | 512            | 2                | bf16-true | 1xA10G  | 12.91 min        | $0.4 | 11.58 GB    | 1.085           | 2.959                 |
| gemma-2b/qlora.yaml               | gemma-2b               | Alpaca2k | 2      | 512            | 2                | bf16-true | 4xA10G  | 12.91 min        | $1.0 | 11.59 GB    | 1.085           | 2.958                 |
|                                   |                        |          |        |                |                  |           |         |                  |      |             |                 |                       |
| gemma-7b/lora.yaml                | gemma-7b               | Alpaca2k | 2      | 512            | 1                | bf16-true | 1xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |
| gemma-7b/lora.yaml                | gemma-7b               | Alpaca2k | 2      | 512            | 1                | bf16-true | 4xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |
| gemma-7b/qlora.yaml               | gemma-7b               | Alpaca2k | 2      | 512            | 1                | bf16-true | 1xA10G  | 43.58 min        | $1.3 | 17.18 GB    | 0.973           | 2.646                 |
| gemma-7b/qlora.yaml               | gemma-7b               | Alpaca2k | 2      | 512            | 1                | bf16-true | 4xA10G  | 43.58 min        | $3.5 | 17.18 GB    | 0.983           | 2.672                 |
|                                   |                        |          |        |                |                  |           |         |                  |      |             |                 |                       |
| llama-2-7b/full.yaml              | llama-2-7b             | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |
| llama-2-7b/lora.yaml              | llama-2-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 32.82 min        | $1.0 | 19.77 GB    | 0.802           | 2.230                 |
| llama-2-7b/lora.yaml              | llama-2-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 32.83 min        | $2.6 | 19.77 GB    | 0.802           | 2.229                 |
| llama-2-7b/qlora.yaml             | llama-2-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 45.67 min        | $1.4 | 13.68 GB    | 0.814           | 2.258                 |
| llama-2-7b/qlora.yaml             | llama-2-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 45.69 min        | $3.7 | 13.68 GB    | 0.815           | 2.258                 |
|                                   |                        |          |        |                |                  |           |         |                  |      |             |                 |                       |
| llama-3-8b/full.yaml              | llama-3-8b             | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |
| llama-3-8b/lora.yaml              | llama-3-8b             | Alpaca2k | 2      | 512            | 1                | bf16-true | 1xA10G  | 14.79 min        | $0.4 | 19.73 GB    | 0.888           | 2.431                 |
| llama-3-8b/lora.yaml              | llama-3-8b             | Alpaca2k | 2      | 512            | 1                | bf16-true | 4xA10G  | 14.88 min        | $1.2 | 19.73 GB    | 0.889           | 2.432                 |
| llama-3-8b/qlora.yaml             | llama-3-8b             | Alpaca2k | 2      | 512            | 2                | bf16-true | 1xA10G  | 22.24 min        | $0.7 | 17.41 GB    | 0.939           | 2.558                 |
| llama-3-8b/qlora.yaml             | llama-3-8b             | Alpaca2k | 2      | 512            | 2                | bf16-true | 4xA10G  | 22.20 min        | $1.8 | 17.41 GB    | 0.939           | 2.557                 |
|                                   |                        |          |        |                |                  |           |         |                  |      |             |                 |                       |
| mistral-7b-v0.2/lora.yaml         | mistral-7b-v0.2        | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 31.00 min        | $0.9 | 20.66 GB    | 0.801           | 2.228                 |
| mistral-7b-v0.2/lora.yaml         | mistral-7b-v0.2        | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 31.00 min        | $2.5 | 20.66 GB    | 0.802           | 2.229                 |
| mistral-7b-v0.2/qlora.yaml        | mistral-7b-v0.2        | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 44.75 min        | $1.3 | 14.29 GB    | 0.813           | 2.255                 |
| mistral-7b-v0.2/qlora.yaml        | mistral-7b-v0.2        | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 44.75 min        | $3.6 | 14.29 GB    | 0.813           | 2.254                 |
|                                   |                        |          |        |                |                  |           |         |                  |      |             |                 |                       |
| mistral-7b/lora.yaml              | mistral-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 31.01 min        | $0.9 | 20.66 GB    | 0.794           | 2.211                 |
| mistral-7b/lora.yaml              | mistral-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 31.03 min        | $2.5 | 20.66 GB    | 0.796           | 2.218                 |
| mistral-7b/qlora.yaml             | mistral-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 44.75 min        | $1.3 | 14.29 GB    | 0.803           | 2.231                 |
| mistral-7b/qlora.yaml             | mistral-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 44.81 min        | $3.6 | 14.29 GB    | 0.803           | 2.233                 |
|                                   |                        |          |        |                |                  |           |         |                  |      |             |                 |                       |
| phi-2/full.yaml                   | phi-2                  | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | 11.87 min        | $1.0 | 14.44 GB    | 1.305           | 3.688                 |
| phi-2/lora.yaml                   | phi-2                  | Alpaca2k | 1      | 512            | 4                | bf16-true | 1xA10G  | 3.78 min         | $0.1 | 13.98 GB    | 0.819           | 2.269                 |
| phi-2/lora.yaml                   | phi-2                  | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | 3.78 min         | $0.3 | 13.98 GB    | 0.820           | 2.271                 |
| phi-2/qlora.yaml                  | phi-2                  | Alpaca2k | 1      | 512            | 4                | bf16-true | 1xA10G  | 4.51 min         | $0.1 | 14.27 GB    | 0.837           | 2.310                 |
| phi-2/qlora.yaml                  | phi-2                  | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | 4.52 min         | $0.4 | 14.27 GB    | 0.837           | 2.309                 |
|                                   |                        |          |        |                |                  |           |         |                  |      |             |                 |                       |
| stablelm-base-alpha-3b/full.yaml  | stablelm-base-alpha-3b | Alpaca2k | 1      | 512            | 1                | bf16-true | 4xA10G  | 70.13 min        | $5.6 | 21.23 GB    | 1.513           | 4.540                 |
| stablelm-base-alpha-3b/lora.yaml  | stablelm-base-alpha-3b | Alpaca2k | 4      | 512            | 1                | bf16-true | 1xA10G  | 13.07 min        | $0.4 | 8.58 GB     | 1.361           | 3.900                 |
| stablelm-base-alpha-3b/lora.yaml  | stablelm-base-alpha-3b | Alpaca2k | 4      | 512            | 1                | bf16-true | 4xA10G  | 13.16 min        | $1.1 | 8.58 GB     | 1.362           | 3.906                 |
| stablelm-base-alpha-3b/qlora.yaml | stablelm-base-alpha-3b | Alpaca2k | 4      | 512            | 1                | bf16-true | 1xA10G  | 25.86 min        | $0.8 | 5.24 GB     | 1.388           | 4.009                 |
| stablelm-base-alpha-3b/qlora.yaml | stablelm-base-alpha-3b | Alpaca2k | 4      | 512            | 1                | bf16-true | 4xA10G  | 25.80 min        | $2.1 | 5.24 GB     | 1.391           | 4.020                 |
|                                   |                        |          |        |                |                  |           |         |                  |      |             |                 |                       |
| tiny-llama/full.yaml              | tiny-llama             | Alpaca2k | 1      | 512            | 4                | bf16-true | 1xA10G  | 2.58 min         | $0.1 | 14.10 GB    | 1.088           | 2.968                 |
| tiny-llama/full.yaml              | tiny-llama             | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | 2.57 min         | $0.2 | 14.10 GB    | 1.088           | 2.968                 |
| tiny-llama/lora.yaml              | tiny-llama             | Alpaca2k | 3      | 512            | 8                | bf16-true | 1xA10G  | 8.09 min         | $0.2 | 13.50 GB    | 1.039           | 2.826                 |
| tiny-llama/qlora.yaml             | tiny-llama             | Alpaca2k | 3      | 512            | 8                | bf16-true | 1xA10G  | 8.70 min         | $0.3 | 16.24 GB    | 1.056           | 2.874                 |
| tiny-llama/qlora.yaml             | tiny-llama             | Alpaca2k | 3      | 512            | 8                | bf16-true | 4xA10G  | 8.70 min         | $0.7 | 16.24 GB    | 1.056           | 2.874                 |

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
