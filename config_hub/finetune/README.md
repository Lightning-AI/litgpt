## Config files

The table below lists the performances you can expect from the provided config files. Note that you can achieve lower memory consumption by lowering the micro batch size as needed. In addition, you can lower the rank (`lora_r`) in the LoRA configuration files and disable LoRA for certain layers (for example, setting `lora_projection` and other LoRA layer-specific parameters to `false`).
For more information, see the [Dealing with out-of-memory (OOM) errors](../../tutorials/oom.md) on lowering the memory requirements.

&nbsp;

| Config                            | Model                  | Dataset  | Epochs | Max seq length | Micro batch size | Precision | Machine | Training runtime | Cost | Peak memory | Validation loss | Validation perplexity |
| --------------------------------- | ---------------------- | -------- | ------ | -------------- | ---------------- | --------- | ------- | ---------------- | ---- | ----------- | --------------- | --------------------- |
| falcon-7b/lora.yaml               | falcon-7b              | Alpaca2k | 4      | 512            | 1                | bf16-true | 1xA10G  | 24.72 min        | 0.7$ | 16.69 GB    | 0.945           | 2.573                 |
| falcon-7b/lora.yaml               | falcon-7b              | Alpaca2k | 4      | 512            | 1                | bf16-true | 4xA10G  | 24.54 min        | 2.0$ | 16.69 GB    | 0.945           | 2.573                 |
| falcon-7b/qlora.yaml              | falcon-7b              | Alpaca2k | 4      | 512            | 1                | bf16-true | 1xA10G  | 50.58 min        | 1.5$ | 9.45 GB     | 0.993           | 2.699                 |
| falcon-7b/qlora.yaml              | falcon-7b              | Alpaca2k | 4      | 512            | 1                | bf16-true | 4xA10G  | 50.60 min        | 4.1$ | 9.45 GB     | 0.993           | 2.700                 |
| gemma-2b/full.yaml                | gemma-2b               | Alpaca2k | 1      | 512            | 1                | bf16-true | 4xA10G  | 13.08 min        | 1.1$ | 17.43 GB    | 1.021           | 2.777                 |
| gemma-2b/lora.yaml                | gemma-2b               | Alpaca2k | 2      | 512            | 2                | bf16-true | 1xA10G  | 9.35 min         | 0.3$ | 12.62 GB    | 0.981           | 2.668                 |
| gemma-2b/lora.yaml                | gemma-2b               | Alpaca2k | 2      | 512            | 2                | bf16-true | 4xA10G  | 9.31 min         | 0.7$ | 12.62 GB    | 0.981           | 2.667                 |
| gemma-2b/qlora.yaml               | gemma-2b               | Alpaca2k | 2      | 512            | 2                | bf16-true | 1xA10G  | 12.84 min        | 0.4$ | 11.59 GB    | 1.084           | 2.957                 |
| gemma-2b/qlora.yaml               | gemma-2b               | Alpaca2k | 2      | 512            | 2                | bf16-true | 4xA10G  | 12.85 min        | 1.0$ | 11.58 GB    | 1.084           | 2.958                 |
| gemma-7b/lora.yaml                | gemma-7b               | Alpaca2k | 2      | 512            | 1                | bf16-true | 1xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |
| gemma-7b/lora.yaml                | gemma-7b               | Alpaca2k | 2      | 512            | 1                | bf16-true | 4xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |
| gemma-7b/qlora.yaml               | gemma-7b               | Alpaca2k | 2      | 512            | 1                | bf16-true | 1xA10G  | 43.36 min        | 1.3$ | 17.18 GB    | 0.977           | 2.657                 |
| gemma-7b/qlora.yaml               | gemma-7b               | Alpaca2k | 2      | 512            | 1                | bf16-true | 4xA10G  | 43.43 min        | 3.5$ | 17.18 GB    | 0.980           | 2.665                 |
| llama-2-7b/full.yaml              | llama-2-7b             | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |
| llama-2-7b/lora.yaml              | llama-2-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 32.72 min        | 1.0$ | 19.77 GB    | 0.802           | 2.230                 |
| llama-2-7b/lora.yaml              | llama-2-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 32.70 min        | 2.6$ | 19.77 GB    | 0.802           | 2.229                 |
| llama-2-7b/qlora.yaml             | llama-2-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 45.52 min        | 1.4$ | 13.67 GB    | 0.814           | 2.258                 |
| llama-2-7b/qlora.yaml             | llama-2-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 45.53 min        | 3.7$ | 13.68 GB    | 0.814           | 2.257                 |
| llama-3-8b/full.yaml              | llama-3-8b             | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | OOM              | OOM  | OOM         | OOM             | OOM                   |
| llama-3-8b/lora.yaml              | llama-3-8b             | Alpaca2k | 2      | 512            | 1                | bf16-true | 1xA10G  | 14.74 min        | 0.4$ | 19.73 GB    | 0.888           | 2.431                 |
| llama-3-8b/lora.yaml              | llama-3-8b             | Alpaca2k | 2      | 512            | 1                | bf16-true | 4xA10G  | 14.70 min        | 1.2$ | 19.73 GB    | 0.888           | 2.431                 |
| llama-3-8b/qlora.yaml             | llama-3-8b             | Alpaca2k | 2      | 512            | 2                | bf16-true | 1xA10G  | 22.15 min        | 0.7$ | 17.41 GB    | 0.939           | 2.558                 |
| llama-3-8b/qlora.yaml             | llama-3-8b             | Alpaca2k | 2      | 512            | 2                | bf16-true | 4xA10G  | 22.14 min        | 1.8$ | 17.41 GB    | 0.939           | 2.558                 |
| mistral-7b-v0.2/lora.yaml         | mistral-7b-v0.2        | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 30.86 min        | 0.9$ | 20.66 GB    | 0.801           | 2.228                 |
| mistral-7b-v0.2/lora.yaml         | mistral-7b-v0.2        | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 30.85 min        | 2.5$ | 20.66 GB    | 0.801           | 2.229                 |
| mistral-7b-v0.2/qlora.yaml        | mistral-7b-v0.2        | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 44.58 min        | 1.3$ | 14.29 GB    | 0.813           | 2.256                 |
| mistral-7b-v0.2/qlora.yaml        | mistral-7b-v0.2        | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 44.58 min        | 3.6$ | 14.29 GB    | 0.813           | 2.254                 |
| mistral-7b/lora.yaml              | mistral-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 30.86 min        | 0.9$ | 20.66 GB    | 0.796           | 2.217                 |
| mistral-7b/lora.yaml              | mistral-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 30.90 min        | 2.5$ | 20.66 GB    | 0.796           | 2.218                 |
| mistral-7b/qlora.yaml             | mistral-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 1xA10G  | 44.62 min        | 1.3$ | 14.29 GB    | 0.803           | 2.233                 |
| mistral-7b/qlora.yaml             | mistral-7b             | Alpaca2k | 4      | 512            | 2                | bf16-true | 4xA10G  | 44.60 min        | 3.6$ | 14.29 GB    | 0.803           | 2.233                 |
| phi-2/full.yaml                   | phi-2                  | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | 12.35 min        | 1.0$ | 14.44 GB    | 1.162           | 3.196                 |
| phi-2/lora.yaml                   | phi-2                  | Alpaca2k | 1      | 512            | 4                | bf16-true | 1xA10G  | 3.78 min         | 0.1$ | 13.98 GB    | 0.812           | 2.252                 |
| phi-2/lora.yaml                   | phi-2                  | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | 3.78 min         | 0.3$ | 13.98 GB    | 0.821           | 2.273                 |
| phi-2/qlora.yaml                  | phi-2                  | Alpaca2k | 1      | 512            | 4                | bf16-true | 1xA10G  | 4.47 min         | 0.1$ | 14.27 GB    | 0.861           | 2.366                 |
| phi-2/qlora.yaml                  | phi-2                  | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | 4.50 min         | 0.4$ | 14.27 GB    | 0.847           | 2.332                 |
| stablelm-base-alpha-3b/full.yaml  | stablelm-base-alpha-3b | Alpaca2k | 1      | 512            | 1                | bf16-true | 4xA10G  | 70.91 min        | 5.7$ | 21.23 GB    | 1.524           | 4.590                 |
| stablelm-base-alpha-3b/lora.yaml  | stablelm-base-alpha-3b | Alpaca2k | 4      | 512            | 1                | bf16-true | 1xA10G  | 12.99 min        | 0.4$ | 8.58 GB     | 1.363           | 3.910                 |
| stablelm-base-alpha-3b/lora.yaml  | stablelm-base-alpha-3b | Alpaca2k | 4      | 512            | 1                | bf16-true | 4xA10G  | 12.99 min        | 1.0$ | 8.58 GB     | 1.368           | 3.929                 |
| stablelm-base-alpha-3b/qlora.yaml | stablelm-base-alpha-3b | Alpaca2k | 4      | 512            | 1                | bf16-true | 1xA10G  | 25.65 min        | 0.8$ | 5.24 GB     | 1.390           | 4.017                 |
| stablelm-base-alpha-3b/qlora.yaml | stablelm-base-alpha-3b | Alpaca2k | 4      | 512            | 1                | bf16-true | 4xA10G  | 25.63 min        | 2.1$ | 5.24 GB     | 1.391           | 4.018                 |
| tiny-llama/full.yaml              | tiny-llama             | Alpaca2k | 1      | 512            | 4                | bf16-true | 1xA10G  | 2.56 min         | 0.1$ | 14.10 GB    | 1.088           | 2.967                 |
| tiny-llama/full.yaml              | tiny-llama             | Alpaca2k | 1      | 512            | 4                | bf16-true | 4xA10G  | 2.56 min         | 0.2$ | 14.10 GB    | 1.087           | 2.967                 |
| tiny-llama/lora.yaml              | tiny-llama             | Alpaca2k | 3      | 512            | 8                | bf16-true | 1xA10G  | 8.06 min         | 0.2$ | 13.50 GB    | 1.039           | 2.825                 |
| tiny-llama/lora.yaml              | tiny-llama             | Alpaca2k | 3      | 512            | 8                | bf16-true | 4xA10G  | 8.06 min         | 0.6$ | 13.50 GB    | 1.038           | 2.824                 |
| tiny-llama/qlora.yaml             | tiny-llama             | Alpaca2k | 3      | 512            | 8                | bf16-true | 1xA10G  | 8.66 min         | 0.3$ | 16.24 GB    | 1.056           | 2.874                 |
| tiny-llama/qlora.yaml             | tiny-llama             | Alpaca2k | 3      | 512            | 8                | bf16-true | 4xA10G  | 8.68 min         | 0.7$ | 16.24 GB    | 1.056           | 2.874                 |

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
