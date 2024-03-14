## Config files

The table below lists the performances you can expect from the provided config files. Note that you can achieve lower memory consumption by lowering the micro batch size as needed. In addition, you can lower the rank (`lora_r`) in the LoRA configuration files and disable LoRA for certain layers (for example, setting `lora_projection` and other LoRA layer-specific parameters to `false`). 
For more information, see the [Dealing with out-of-memory (OOM) errors](../../tutorials/oom.md) on lowering the memory requirements.

|                       | Size | Dataset   | Epochs | Val loss | Peak memory | Max seq length | Micro batch size | Precision | Training runtime |
| --------------------- | ---- | --------- | ------ | -------- | ----------- | -------------- | ---------------- | --------- | ---------------- |
| tiny-llama/lora.yaml  | 1.1B | Alpaca 2k | 3      | 1.038    | 13.50 GB    | 512            | 8                | bfloat16  | 8.06 min (A10G)  |
| tiny-llama/qlora.yaml | 1.1B | Alpaca 2k | 3      | 1.056    | 16.24 GB    | 512            | 8                | bfloat16  | 8.74 min (A10G)  |
| tiny-llama/full.yaml  | 1.1B | Alpaca 2k | 1      | 1.105    | 14.10 GB    | 512            | 4                | bfloat16  | 2.59 min (A10G)  |
| llama-2-7b/qlora.yaml | 7B   | Alpaca 2k | 4      | 0.814    | 13.68 GB    | 512            | 2                | bfloat16  | 45.68 min (A10G) |
