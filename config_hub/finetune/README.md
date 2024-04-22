## Config files

The table below lists the performances you can expect from the provided config files. Note that you can achieve lower memory consumption by lowering the micro batch size as needed. In addition, you can lower the rank (`lora_r`) in the LoRA configuration files and disable LoRA for certain layers (for example, setting `lora_projection` and other LoRA layer-specific parameters to `false`).
For more information, see the [Dealing with out-of-memory (OOM) errors](../../tutorials/oom.md) on lowering the memory requirements.

&nbsp;

| Config | Model | Dataset | Epochs | Max seq length | Micro batch size | Precision | Machine | Training runtime | Cost | Peak memory | Validation loss | Validation perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| falcon-7b/lora.yaml | falcon-7b | Alpaca2k | 4 | 512 | 1 | bf16-true | 4xA10G | 24.64 min | 2.0$ | 16.69 GB | 0.9448 | 2.5723 |
| falcon-7b/qlora.yaml | falcon-7b | Alpaca2k | 4 | 512 | 1 | bf16-true | 1xA10G | 50.60 min | 4.1$ | 9.45 GB | 0.9930 | 2.6992 |
| falcon-7b/qlora.yaml | falcon-7b | Alpaca2k | 4 | 512 | 1 | bf16-true | 4xA10G | 50.60 min | 4.1$ | 9.45 GB | 0.9931 | 2.6996 |
| gemma-2b/lora.yaml | gemma-2b | Alpaca2k | 2 | 512 | 2 | bf16-true | 4xA10G | 9.29 min | 0.7$ | 12.62 GB | 0.9809 | 2.6669 |
| gemma-2b/qlora.yaml | gemma-2b | Alpaca2k | 2 | 512 | 2 | bf16-true | 1xA10G | 12.79 min | 1.0$ | 11.57 GB | 1.0846 | 2.9582 |
| gemma-2b/qlora.yaml | gemma-2b | Alpaca2k | 2 | 512 | 2 | bf16-true | 4xA10G | 12.82 min | 1.0$ | 11.57 GB | 1.0845 | 2.9580 |
| gemma-7b/qlora.yaml | gemma-7b | Alpaca2k | 2 | 512 | 1 | bf16-true | 1xA10G | 43.35 min | 3.5$ | 17.18 GB | 0.9754 | 2.6522 |
| gemma-7b/qlora.yaml | gemma-7b | Alpaca2k | 2 | 512 | 1 | bf16-true | 4xA10G | 43.42 min | 3.5$ | 17.18 GB | 0.9822 | 2.6704 |
| llama-2-7b/lora.yaml | llama-2-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 4xA10G | 32.68 min | 2.6$ | 19.77 GB | 0.8017 | 2.2294 |
| llama-2-7b/qlora.yaml | llama-2-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 1xA10G | 45.54 min | 3.7$ | 13.67 GB | 0.8142 | 2.2573 |
| llama-2-7b/qlora.yaml | llama-2-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 4xA10G | 45.56 min | 3.7$ | 13.68 GB | 0.8141 | 2.2571 |
| llama-3-8b/lora.yaml | llama-3-8b | Alpaca2k | 2 | 512 | 1 | bf16-true | 4xA10G | 14.75 min | 1.2$ | 19.73 GB | 0.8885 | 2.4316 |
| llama-3-8b/qlora.yaml | llama-3-8b | Alpaca2k | 2 | 512 | 2 | bf16-true | 1xA10G | 22.17 min | 1.8$ | 17.41 GB | 0.9388 | 2.5568 |
| llama-3-8b/qlora.yaml | llama-3-8b | Alpaca2k | 2 | 512 | 2 | bf16-true | 4xA10G | 22.13 min | 1.8$ | 17.41 GB | 0.9389 | 2.5573 |
| mistral-7b/lora.yaml | mistral-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 4xA10G | 30.85 min | 2.5$ | 20.66 GB | 0.7927 | 2.2092 |
| mistral-7b/qlora.yaml | mistral-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 1xA10G | 44.62 min | 3.6$ | 14.29 GB | 0.8030 | 2.2322 |
| mistral-7b/qlora.yaml | mistral-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 4xA10G | 44.64 min | 3.6$ | 14.29 GB | 0.8031 | 2.2323 |
| phi-2/lora.yaml | phi-2 | Alpaca2k | 1 | 512 | 4 | bf16-true | 4xA10G | 3.77 min | 0.3$ | 13.98 GB | 0.8191 | 2.2685 |
| phi-2/qlora.yaml | phi-2 | Alpaca2k | 1 | 512 | 4 | bf16-true | 1xA10G | 4.48 min | 0.4$ | 14.27 GB | 0.8448 | 2.3275 |
| phi-2/qlora.yaml | phi-2 | Alpaca2k | 1 | 512 | 4 | bf16-true | 4xA10G | 4.49 min | 0.4$ | 14.27 GB | 0.8600 | 2.3632 |
| stablelm-base-alpha-3b/lora.yaml | stablelm-base-alpha-3b | Alpaca2k | 4 | 512 | 1 | bf16-true | 4xA10G | 12.99 min | 1.0$ | 8.58 GB | 1.3640 | 3.9119 |
| stablelm-base-alpha-3b/qlora.yaml | stablelm-base-alpha-3b | Alpaca2k | 4 | 512 | 1 | bf16-true | 1xA10G | 25.59 min | 2.1$ | 5.24 GB | 1.3907 | 4.0177 |
| stablelm-base-alpha-3b/qlora.yaml | stablelm-base-alpha-3b | Alpaca2k | 4 | 512 | 1 | bf16-true | 4xA10G | 25.66 min | 2.1$ | 5.24 GB | 1.3913 | 4.0199 |
| tiny-llama/full.yaml | tiny-llama | Alpaca2k | 1 | 512 | 4 | bf16-true | 4xA10G | 2.56 min | 0.2$ | 14.10 GB | 1.0889 | 2.9711 |
| tiny-llama/lora.yaml | tiny-llama | Alpaca2k | 3 | 512 | 8 | bf16-true | 4xA10G | 8.07 min | 0.6$ | 13.50 GB | 1.0385 | 2.8251 |
| tiny-llama/qlora.yaml | tiny-llama | Alpaca2k | 3 | 512 | 8 | bf16-true | 1xA10G | 8.67 min | 0.7$ | 16.24 GB | 1.0560 | 2.8747 |
| tiny-llama/qlora.yaml | tiny-llama | Alpaca2k | 3 | 512 | 8 | bf16-true | 4xA10G | 8.67 min | 0.7$ | 16.24 GB | 1.0560 | 2.8748 |

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
