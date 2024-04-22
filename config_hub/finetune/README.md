## Config files

The table below lists the performances you can expect from the provided config files. Note that you can achieve lower memory consumption by lowering the micro batch size as needed. In addition, you can lower the rank (`lora_r`) in the LoRA configuration files and disable LoRA for certain layers (for example, setting `lora_projection` and other LoRA layer-specific parameters to `false`).
For more information, see the [Dealing with out-of-memory (OOM) errors](../../tutorials/oom.md) on lowering the memory requirements.

&nbsp;

| Config | Model | Dataset | Epochs | Max seq length | Micro batch size | Precision | Machine | Training runtime | Cost | Peak memory | Validation loss | Validation perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| falcon-7b/lora.yaml | falcon-7b | Alpaca2k | 4 | 512 | 1 | bf16-true | 1xA10G | 24.58 min | 0.7$ | 16.69 GB | 0.9452 | 2.5732 |
| falcon-7b/lora.yaml | falcon-7b | Alpaca2k | 4 | 512 | 1 | bf16-true | 4xA10G | 24.67 min | 2.0$ | 16.69 GB | 0.9450 | 2.5727 |
| falcon-7b/qlora.yaml | falcon-7b | Alpaca2k | 4 | 512 | 1 | bf16-true | 1xA10G | 50.56 min | 1.5$ | 9.45 GB | 0.9929 | 2.6992 |
| falcon-7b/qlora.yaml | falcon-7b | Alpaca2k | 4 | 512 | 1 | bf16-true | 4xA10G | 50.58 min | 4.1$ | 9.44 GB | 0.9933 | 2.7000 |
| gemma-2b/full.yaml | gemma-2b | Alpaca2k | 1 | 512 | 1 | bf16-true | 4xA10G | 14.03 min | 1.1$ | 17.43 GB | 1.0216 | 2.7775 |
| gemma-2b/lora.yaml | gemma-2b | Alpaca2k | 2 | 512 | 2 | bf16-true | 1xA10G | 9.29 min | 0.3$ | 12.62 GB | 0.9809 | 2.6667 |
| gemma-2b/lora.yaml | gemma-2b | Alpaca2k | 2 | 512 | 2 | bf16-true | 4xA10G | 9.34 min | 0.8$ | 12.62 GB | 0.9806 | 2.6660 |
| gemma-2b/qlora.yaml | gemma-2b | Alpaca2k | 2 | 512 | 2 | bf16-true | 1xA10G | 12.87 min | 0.4$ | 11.58 GB | 1.0847 | 2.9584 |
| gemma-2b/qlora.yaml | gemma-2b | Alpaca2k | 2 | 512 | 2 | bf16-true | 4xA10G | 12.81 min | 1.0$ | 11.59 GB | 1.0847 | 2.9586 |
| gemma-7b/lora.yaml | gemma-7b | Alpaca2k | 2 | 512 | 1 | bf16-true | 1xA10G | OOM | OOM | OOM | OOM | OOM |
| gemma-7b/lora.yaml | gemma-7b | Alpaca2k | 2 | 512 | 1 | bf16-true | 4xA10G | OOM | OOM | OOM | OOM | OOM |
| gemma-7b/qlora.yaml | gemma-7b | Alpaca2k | 2 | 512 | 1 | bf16-true | 1xA10G | 43.37 min | 1.3$ | 17.18 GB | 0.9784 | 2.6603 |
| gemma-7b/qlora.yaml | gemma-7b | Alpaca2k | 2 | 512 | 1 | bf16-true | 4xA10G | 43.38 min | 3.5$ | 17.18 GB | 0.9754 | 2.6521 |
| llama-2-7b/full.yaml | llama-2-7b | Alpaca2k | 1 | 512 | 4 | bf16-true | 4xA10G | OOM | OOM | OOM | OOM | OOM |
| llama-2-7b/lora.yaml | llama-2-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 1xA10G | 32.67 min | 1.0$ | 19.77 GB | 0.8023 | 2.2306 |
| llama-2-7b/lora.yaml | llama-2-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 4xA10G | 32.68 min | 2.6$ | 19.77 GB | 0.8021 | 2.2301 |
| llama-2-7b/qlora.yaml | llama-2-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 1xA10G | 45.52 min | 1.4$ | 13.68 GB | 0.8142 | 2.2574 |
| llama-2-7b/qlora.yaml | llama-2-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 4xA10G | 45.51 min | 3.7$ | 13.68 GB | 0.8144 | 2.2578 |
| llama-3-8b/full.yaml | llama-3-8b | Alpaca2k | 1 | 512 | 4 | bf16-true | 4xA10G | OOM | OOM | OOM | OOM | OOM |
| llama-3-8b/lora.yaml | llama-3-8b | Alpaca2k | 2 | 512 | 1 | bf16-true | 1xA10G | 14.57 min | 0.4$ | 19.73 GB | 0.8882 | 2.4309 |
| llama-3-8b/lora.yaml | llama-3-8b | Alpaca2k | 2 | 512 | 1 | bf16-true | 4xA10G | 14.72 min | 1.2$ | 19.73 GB | 0.8885 | 2.4315 |
| llama-3-8b/qlora.yaml | llama-3-8b | Alpaca2k | 2 | 512 | 2 | bf16-true | 1xA10G | 22.17 min | 0.7$ | 17.41 GB | 0.9392 | 2.5579 |
| llama-3-8b/qlora.yaml | llama-3-8b | Alpaca2k | 2 | 512 | 2 | bf16-true | 4xA10G | 22.12 min | 1.8$ | 17.41 GB | 0.9393 | 2.5582 |
| mistral-7b-v0.2/lora.yaml | mistral-7b-v0.2 | Alpaca2k | 4 | 512 | 2 | bf16-true | 1xA10G | 30.90 min | 0.9$ | 20.66 GB | 0.8020 | 2.2299 |
| mistral-7b-v0.2/lora.yaml | mistral-7b-v0.2 | Alpaca2k | 4 | 512 | 2 | bf16-true | 4xA10G | 30.87 min | 2.5$ | 20.66 GB | 0.8014 | 2.2286 |
| mistral-7b-v0.2/qlora.yaml | mistral-7b-v0.2 | Alpaca2k | 4 | 512 | 2 | bf16-true | 1xA10G | 44.56 min | 1.3$ | 14.29 GB | 0.8138 | 2.2564 |
| mistral-7b-v0.2/qlora.yaml | mistral-7b-v0.2 | Alpaca2k | 4 | 512 | 2 | bf16-true | 4xA10G | 44.57 min | 3.6$ | 14.29 GB | 0.8132 | 2.2551 |
| mistral-7b/lora.yaml | mistral-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 1xA10G | 30.92 min | 0.9$ | 20.66 GB | 0.7953 | 2.2151 |
| mistral-7b/lora.yaml | mistral-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 4xA10G | 30.95 min | 2.5$ | 20.66 GB | 0.7962 | 2.2170 |
| mistral-7b/qlora.yaml | mistral-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 1xA10G | 44.62 min | 1.3$ | 14.29 GB | 0.8031 | 2.2324 |
| mistral-7b/qlora.yaml | mistral-7b | Alpaca2k | 4 | 512 | 2 | bf16-true | 4xA10G | 44.63 min | 3.6$ | 14.29 GB | 0.8030 | 2.2322 |
| phi-2/full.yaml | phi-2 | Alpaca2k | 1 | 512 | 4 | bf16-true | 4xA10G | 11.97 min | 1.0$ | 14.44 GB | 1.1616 | 3.1949 |
| phi-2/lora.yaml | phi-2 | Alpaca2k | 1 | 512 | 4 | bf16-true | 1xA10G | 3.78 min | 0.1$ | 13.98 GB | 0.8101 | 2.2482 |
| phi-2/lora.yaml | phi-2 | Alpaca2k | 1 | 512 | 4 | bf16-true | 4xA10G | 3.79 min | 0.3$ | 13.98 GB | 0.8098 | 2.2475 |
| phi-2/qlora.yaml | phi-2 | Alpaca2k | 1 | 512 | 4 | bf16-true | 1xA10G | 4.49 min | 0.1$ | 14.27 GB | 0.8413 | 2.3194 |
| phi-2/qlora.yaml | phi-2 | Alpaca2k | 1 | 512 | 4 | bf16-true | 4xA10G | 4.51 min | 0.4$ | 14.27 GB | 0.8527 | 2.3459 |
| stablelm-base-alpha-3b/full.yaml | stablelm-base-alpha-3b | Alpaca2k | 1 | 512 | 1 | bf16-true | 4xA10G | 66.47 min | 5.4$ | 21.23 GB | 1.5193 | 4.5691 |
| stablelm-base-alpha-3b/lora.yaml | stablelm-base-alpha-3b | Alpaca2k | 4 | 512 | 1 | bf16-true | 1xA10G | 12.88 min | 0.4$ | 8.58 GB | 1.3613 | 3.9013 |
| stablelm-base-alpha-3b/lora.yaml | stablelm-base-alpha-3b | Alpaca2k | 4 | 512 | 1 | bf16-true | 4xA10G | 12.91 min | 1.0$ | 8.58 GB | 1.3609 | 3.8999 |
| stablelm-base-alpha-3b/qlora.yaml | stablelm-base-alpha-3b | Alpaca2k | 4 | 512 | 1 | bf16-true | 1xA10G | 25.62 min | 0.8$ | 5.24 GB | 1.3914 | 4.0204 |
| stablelm-base-alpha-3b/qlora.yaml | stablelm-base-alpha-3b | Alpaca2k | 4 | 512 | 1 | bf16-true | 4xA10G | 25.72 min | 2.1$ | 5.24 GB | 1.3918 | 4.0222 |
| tiny-llama/full.yaml | tiny-llama | Alpaca2k | 1 | 512 | 4 | bf16-true | 1xA10G | 2.56 min | 0.1$ | 14.10 GB | 1.0877 | 2.9676 |
| tiny-llama/full.yaml | tiny-llama | Alpaca2k | 1 | 512 | 4 | bf16-true | 4xA10G | 2.57 min | 0.2$ | 14.10 GB | 1.0881 | 2.9686 |
| tiny-llama/lora.yaml | tiny-llama | Alpaca2k | 3 | 512 | 8 | bf16-true | 1xA10G | 8.06 min | 0.2$ | 13.50 GB | 1.0383 | 2.8244 |
| tiny-llama/lora.yaml | tiny-llama | Alpaca2k | 3 | 512 | 8 | bf16-true | 4xA10G | 8.05 min | 0.6$ | 13.50 GB | 1.0382 | 2.8242 |
| tiny-llama/qlora.yaml | tiny-llama | Alpaca2k | 3 | 512 | 8 | bf16-true | 1xA10G | 8.66 min | 0.3$ | 16.24 GB | 1.0559 | 2.8744 |
| tiny-llama/qlora.yaml | tiny-llama | Alpaca2k | 3 | 512 | 8 | bf16-true | 4xA10G | 8.67 min | 0.7$ | 16.24 GB | 1.0560 | 2.8747 |

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
