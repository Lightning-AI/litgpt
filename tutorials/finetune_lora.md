# Finetuning with LoRA / QLoRA

[Low-rank adaption (LoRA)](https://arxiv.org/abs/2106.09685) is a technique to approximate the update to the linear layers in a LLM with a low-rank matrix factorization. This significantly reduces the number of trainable parameters and speeds up training with little impact on the final performance of the model.
We demonstrate this method by instruction-finetuning Lit-GPT StableLM 3B on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset on a **single RTX 3090 (24GB) GPU**.

## Preparation

The steps here only need to be done once:

1. Follow the instructions in the [README](../README.md) to install the dependencies.
2. Download and convert the weights and save them in the `./checkpoints` folder.
   Weights can be downloaded following these instructions:

- [StableLM](download_stablelm.md)
- [Pythia](download_pythia.md)
- [Redpajama-INCITE](download_redpajama_incite.md)
- [Falcon](download_falcon.md)

3. Download the data and generate the instruction tuning dataset:

   ```bash
   python scripts/prepare_alpaca.py
   ```

(See [this blog article](https://lightning.ai/blog/how-to-finetune-gpt-like-large-language-models-on-a-custom-dataset) for how to prepare and use custom datasets.)

## Running the finetuning

```bash
python finetune/lora.py
```

The finetuning requires at least one GPU with ~24 GB memory (RTX 3090).

This script will save checkpoints periodically to the folder `out/`.

> [!NOTE]\: LoRA can be applied to not only `query`, `key` or `value` matrices, but also to `projection`, `mlp` and classification `head`.
> According to [QLoRA](https://arxiv.org/abs/2305.14314) paper (section 4): "LoRA on all linear transformer block layers are required to match full finetuning performance".
> By default LoRA is applied only to the `query` and `value` matrices. In order to apply LoRA to other weight matrices - change the variables in `finetune/lora.py` accordingly.

Optionally, finetuning using 4-bit quantization (as in QLoRA) can be enabled via the `--quantize` flag, for example using the 4-bit NormalFloat data type:

```bash
python finetune/lora.py --quantize "bnb.nf4"
```

and optionally with double-quantization:

```bash
python finetune/lora.py --quantize "bnb.nf4-dq"
```

The table below lists a comparison with different settings on a StableLM 3B model finetuned with LoRA on Alpaca for 5,000 iterations using a microbatch size of 4:

| Settings                                                | Training Memory  | Training Time  | Loss      | Inference Memory |
|---------------------------------------------------------|------------------|----------------|-----------|------------------|
| Default (bfloat16-mixed)                                | 33.50 GB         | 591.78s        | 0.9207    | 7.61 GB          |
| --precision "bf16-true"                                 | 15.86 GB         | 592.14s        | 0.9180    | 7.61 GB          |
| --quantize "bnb.nf4"                                    | 22.34 GB         | 944.93s        | 0.9417    | 3.25 GB          |
| --quantize "bnb.nf4-dq"                                 | 22.18 GB         | 962.23s        | 0.9383    | 3.08 GB          |
| --precision "bf16-true"  --quantize "bnb.nf4"           | 14.81 GB         | 802.02s        | 0.9408    | 3.25 GB          |
| --precision "bf16-true"  --quantize "bnb.nf4-dq"        | 14.65 GB         | 802.94s        | 0.9384    | 3.08 GB          |

The advantages of QLoRA-style quantization are more pronounced in larger models, such as Llama 2 7B. The table below summarizes the results for Llama 2 7B on Alpaca for 5,000 iterations using a microbatch size of 4:

| Settings                                            | Training Memory  | Training Time  | Loss   | Inference Memory |
|-----------------------------------------------------|------------------|----------------|--------|------------------|
| Default (bfloat16-mixed)                            | OutOfMemoryError | N/A            | N/A    | N/A              |
| --precision "bf16-true"                             | 20.60 GB         | 876.30s        | 0.8696 | 13.82 GB         |
| --quantize "bnb.nf4"                                | 19.62 GB         | 1320.63s       | 1.0178 | 4.66 GB          |
| --quantize "bnb.nf4-dq"                             | 19.32 GB         | 1359.10s       | 1.0132 | 4.34 GB          |
| --precision "bf16-true"  --quantize "bnb.nf4"       | 13.44 GB         | 1089.79s       | 1.0130 | 4.66 GB          |
| --precision "bf16-true"  --quantize "bnb.nf4-dq"    | 13.15 GB         | 1135.86s       | 1.0124 | 4.34 GB          |

## Test the model

You can test the finetuned model with your own instructions by running:

```bash
python generate/lora.py --prompt "Recommend a movie to watch on the weekend."
```

Output:

```text
I would recommend the movie The Martian (2015). It is a sci-fi movie starring Matt Damon that follows the story of...
```

If your GPU supports `bfloat16`, you can additionally pass `--precision "bf16-true"` to bring the memory consumption down to ~7.6 GB for StableLM-3B (versus ~15.2  GB for `--precision "32-full"`). In addition, you may use quantization methods, for example `--precision "bf16-true" --quantize "bnb.nf4"` brings the memory consumption further down to ~4.4 GB for StableLM-3B.

## Tune on your dataset

With only a few modifications, you can prepare and train on your own instruction dataset.

1. Create a json file in which each row holds one instruction-response pair.
   A row has an entry for 'instruction', 'input', and 'output', where 'input' is optional an can be
   the empty string if the instruction doesn't require a context. Below is an example json file:

   ```text
   [
       {
           "instruction": "Arrange the given numbers in ascending order.",
           "input": "2, 4, 0, 8, 3",
           "output": "0, 2, 3, 4, 8"
       },
       ...
   ]
   ```

2. Make a copy of `scripts/prepare_alpaca.py` and name it what you want:

   ```bash
   cp scripts/prepare_alpaca.py scripts/prepare_mydata.py
   ```

3. Modify `scripts/prepare_mydata.py` to read the json data file.
4. Run the script to generate the preprocessed, tokenized train-val split:

   ```bash
   python scripts/prepare_mydata.py --destination_path data/mydata/
   ```

5. Run `finetune/lora.py` by passing in the location of your data (and optionally other parameters):

   ```bash
   python finetune/lora.py --data_dir data/mydata/ --out_dir out/myexperiment
   ```

## Troubleshooting

If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see <https://github.com/Lightning-AI/lit-llama/issues/101>).
