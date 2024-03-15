## Config files

The table below lists the performances you can expect from the provided config files. Note that you can achieve lower memory consumption by lowering the micro batch size as needed. In addition, you can lower the rank (`lora_r`) in the LoRA configuration files and disable LoRA for certain layers (for example, setting `lora_projection` and other LoRA layer-specific parameters to `false`). 
For more information, see the [Dealing with out-of-memory (OOM) errors](../../tutorials/oom.md) on lowering the memory requirements.

&nbsp;

|                       | Size | Dataset   | Epochs | Val loss | Peak memory | Max seq length | Micro batch size | Precision | Training runtime   |
| --------------------- | ---- | --------- | ------ | -------- | ----------- | -------------- | ---------------- | --------- | -------------------|
| gemma-2b/lora.yaml    | 2B   | Alpaca 2k | 3      | 1.476    | 12.62 GB    | 512            | 2                | bfloat16  | 18.31 min (1xA10G) |
| gemma-2b/qlora.yaml   | 2B   | Alpaca 2k | 3      | 1.626    | 11.51 GB    | 512            | 2                | bfloat16  | 25.29 min (1xA10G) |
| gemma-2b/full.yaml    | 2B   | Alpaca 2k | 0.35   | 1.046    | 18.47 GB    | 512            | 2                | bfloat16  | 16.79 min (2xA10G) |
|                       |      |           |        |          |             |                |                  |           |                    |
| llama-2-7b/qlora.yaml | 7B   | Alpaca 2k | 4      | 0.814    | 13.68 GB    | 512            | 2                | bfloat16  | 45.68 min (1xA10G) |
|                       |      |           |        |          |             |                |                  |           |                    |
| phi-2/lora.yaml       | 2B   | Alpaca 2k | 1      | 0.832    | 13.98 GB    | 512            | 4                | bfloat16  | 3.82 min (1xA10G)  |
| phi-2/qlora.yaml      | 2B   | Alpaca 2k | 1      | 0.846    | 14.27 GB    | 512            | 4                | bfloat16  | 4.55 min (1xA10G)  |
| phi-2/full.yaml       | 2B   | Alpaca 2k | 1      | 0.937    | 14.44 GB    | 512            | 4                | bfloat16  | 13.00 min (1xA10G) |
|                       |      |           |        |          |             |                |                  |           |                    |
| tiny-llama/lora.yaml  | 1.1B | Alpaca 2k | 3      | 1.038    | 13.50 GB    | 512            | 8                | bfloat16  | 8.06 min (1xA10G)  |
| tiny-llama/qlora.yaml | 1.1B | Alpaca 2k | 3      | 1.056    | 16.24 GB    | 512            | 8                | bfloat16  | 8.74 min (1xA10G)  |
| tiny-llama/full.yaml  | 1.1B | Alpaca 2k | 1      | 1.105    | 14.10 GB    | 512            | 4                | bfloat16  | 2.59 min (1xA10G)  |

&nbsp;

If you require a longer sequence length than the one used in a given config file, you can either edit the `max_seq_length` in the config file or pass an additional argument when running the finetuning command, for example, `--max_seq_length 4096` to override the sequence length provided in the config file.
