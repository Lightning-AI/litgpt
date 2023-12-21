# Finetuning with LoRA / QLoRA

[Low-rank adaption (LoRA)](https://arxiv.org/abs/2106.09685) is a technique to approximate the update to the linear layers in a LLM with a low-rank matrix factorization. This significantly reduces the number of trainable parameters and speeds up training with little impact on the final performance of the model.
We demonstrate this method by instruction-finetuning Lit-GPT StableLM 3B on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset on a **single RTX 3090 (24GB) GPU** with CUDA 11.8.

&nbsp;

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
python scripts/prepare_alpaca.py \
  --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

or [prepare your own dataset](#tune-on-your-dataset).

For more information about dataset preparation, also see the [prepare_dataset.md](./prepare_dataset.md) tutorial.

&nbsp;

## Running the Finetuning

```bash
python finetune/lora.py
```

The finetuning requires at least one GPU with ~24 GB memory (RTX 3090).

This script will save checkpoints periodically to the folder `out/`.

> [!NOTE]
> LoRA can be applied to not only `query`, `key` or `value` matrices, but also to `projection`, `mlp` and classification `head`.
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

The table below lists a comparison with different settings on a StableLM 3B model finetuned with LoRA on Alpaca for 1,000 iterations using a microbatch size of 1:

| Settings                                    | Training Memory | Training Time |  Inference Memory |
|---------------------------------------------|-----------------|---------------|-------------------|
| Default (bf16-mixed)                        | 26.92 GB        | 1.34 min      | 21.43 GB          |
| --precision bf16-true                       | 9.69 GB         | 1.24 min      | 7.30 GB           |
| --precision bf16-true --quantize bnb.nf4    | 6.35 GB         | 1.82 min      | 3.20 GB           |
| --precision bf16-true --quantize bnb.nf4-dq | 6.19 GB         | 1.87 min      | 3.04 GB           |

The advantages of QLoRA-style quantization are more pronounced in larger models, such as Llama 2 7B. The table below summarizes the results for Llama 2 7B on Alpaca for 1,000 iterations using a microbatch size of 1:

| Settings                                    | Training Memory  | Training Time | Inference Memory |
|---------------------------------------------|------------------|---------------|------------------|
| Default (bf16-mixed)                        | OutOfMemoryError | N/A           | 40.21 GB         |
| --precision bf16-true                       | 21.30 GB         | 2.36 min      | 13.52 GB         |
| --precision bf16-true --quantize bnb.nf4    | 14.14 GB         | 3.68 min      | 4.57 GB          |
| --precision bf16-true --quantize bnb.nf4-dq | 13.84 GB         | 3.83 min      | 4.26 GB          |

For additional benchmarks and resource requirements, please see the [Resource Tables](resource-tables.md).

&nbsp;

## Test the Model

You can test the finetuned model with your own instructions by running:

```bash
python generate/lora.py \
  --prompt "Recommend a movie to watch on the weekend."
```

Output:

```text
I would recommend the movie The Martian (2015). It is a sci-fi movie starring Matt Damon that follows the story of...
```

If your GPU supports `bfloat16`, you can additionally pass `--precision "bf16-true"` to bring the memory consumption down to ~7.6 GB for StableLM-3B (versus ~15.2  GB for `--precision "32-full"`). In addition, you may use quantization methods, for example `--precision "bf16-true" --quantize "bnb.nf4"` brings the memory consumption further down to ~4.4 GB for StableLM-3B.

&nbsp;

## Tune on Your Dataset

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
   python scripts/prepare_mydata.py \
     --destination_path data/mydata/
   ```

5. Run `finetune/lora.py` by passing in the location of your data (and optionally other parameters):

   ```bash
   python finetune/lora.py  \
     --data_dir data/mydata/ \
     --out_dir out/myexperiment
   ```

&nbsp;

## Merging LoRA Weights

By default, the LoRA weights are kept separate from the checkpoint file to save storage space.
However, you can optionally merge the LoRA weights with the original model checkpoint to create
a new file to optimize inference speeds. (This will improve inference performance
because the weights don't have to be added during runtime.)

Let's assume we finetuned a model using LoRA as follows:

```bash
python finetune/lora.py \
  --checkpoint_dir "checkpoints/stabilityai/stablelm-base-alpha-3b/" \
  --data_dir "data/alpaca" \
  --out_dir "out/lora_weights/stablelm-base-alpha-3b/"
```

Then, we can merge the LoRA weights with the checkpoint model using the `merge_lora.py` script as shown below:

```bash
python scripts/merge_lora.py \
  --checkpoint_dir "checkpoints/stabilityai/stablelm-base-alpha-3b/" \
  --lora_path "out/lora_weights/stablelm-base-alpha-3b/lit_model_lora_finetuned.pth" \
  --out_dir "out/lora_merged/stablelm-base-alpha-3b/"
```

> [!IMPORTANT]
> If you changed the LoRA hyperparameters (`lora_r`, `lora_key`, etc.) in the
> `finetune/lora.py` script, it is important to update the hyperparameter configuration
> in the `scripts/merge_lora.py` script accordingly. Otherwise, you will encounter size
> mismatch errors upon merging.

After merging, we can use the `base.py` file for inference using the new checkpoint file. Note that if your new checkpoint directory is different from the original checkpoint directory, we also have to copy over the tokenizer and config files:

```bash
cp checkpoints/stabilityai/stablelm-base-alpha-3b/*.json \
out/lora_merged/stablelm-base-alpha-3b/
```

> [!Note]
> Some models (for example, Llama 2) also come with a `tokenizer.model` file.
> In this case, you also need to use an additional copy step:
> `cp checkpoints/origin/tokenizer.model out/lora_merged/target/`

Then, we should be ready to use the model in inference:

```bash
python generate/base.py \
  --checkpoint_dir "out/lora_merged/stablelm-base-alpha-3b/"
```

Similarly, you can evaluate the model using the `eval/lm_eval_harness.py` script (see the [evaluation](evaluation.md) tutorial for more information):

```bash
python eval/lm_eval_harness.py \
  --checkpoint_dir "out/lora_merged/stablelm-base-alpha-3b/" \
  --precision "bf16-true" \
  --save_filepath "results.json"
```
