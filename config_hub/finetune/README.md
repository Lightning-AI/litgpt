## Config files

The table below lists the performances you can expect from the provided config files. Note that you can achieve lower memory consumption by lowering the micro batch size as needed. See the [Dealing with out-of-memory (OOM) errors](../../litgpt/tutorials/oom.md) on lowering the memory requirements.

|                       | Size | Dataset   | Epochs | Val loss | Peak memory | Max seq length | Micro batch size | Precision | Training runtime |
| --------------------- | ---- | --------- | ------ | -------- | ----------- | -------------- | ---------------- | --------- | ---------------- |
| tiny-llama/lora.yaml  | 1.1B | Alpaca 2k | 1      | 1.053    | 10.54 GB    | 512            | 8                | bfloat16  | 9.24 min (A10G)  |
| tiny-llama/qlora.yaml | 1.1B | Alpaca 2k | 4      | 1.074    | 13.32 GB    | 512            | 8                | bfloat16  | 9.89 min (A10G)  |
| tiny-llama/full.yaml  | 1.1B | Alpaca 2k | 4      | 1.105    | 14.10 GB    | 512            | 4                | bfloat16  | 2.59 min (A10G)  |
| llama-2-7b/qlora.yaml | 7B   | Alpaca 2k | 4      | 0.814    | 13.68 GB     | 512            | 2                | bfloat16  | 45.68 min (A10G) |
