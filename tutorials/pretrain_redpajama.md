# Pretrain Llama 2 on RedPajama

This tutorial will walk you through setting up the RedPajama dataset and launching the pretraining script.

## What's RedPajama

[RedPajama](https://github.com/togethercomputer/RedPajama-Data) is an open-source reproduction of the original LLaMA training dataset.

It contains a total of 1.2 trillion tokens, divided into

| Name          | Size |
|---------------|------|
| Commoncrawl   | 878B |
| C4            | 175B |
| GitHub        | 59B  |
| Books         | 26B  |
| ArXiv         | 28B  |
| Wikipedia     | 24B  |
| StackExchange | 20B  |

The [RedPajama repo](https://github.com/togethercomputer/RedPajama-Data) contains the source code for collecting and preparing the dataset, which is Apache 2.0 licensed.

The data itself is licensed according to the original licenses with which its individual parts were released.
The GitHub datasets are limited to MIT, BSD, or Apache 2.0 repositories.

Along with the full [RedPajama-1T dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T),
the smaller [RedPajama-1T-Sample](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) 1B sample dataset is also available for development.

You can download the data using git lfs:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com):
sudo apt install git-lfs
```

```bash
# The full 1 trillion token dataset:
git clone https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T data/RedPajama-Data-1T
```

```bash
# The 1 billion token subset
git clone https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \
  data/RedPajama-Data-1T-Sample
```

## Prepare RedPajama for training

The full dataset consists of 2084 `jsonl` files (the sample dataset contains 11). In order to start pretraining lit-gpt
on it, you need to read, tokenize, and write the data in binary chunks. This will leverage the `PackedDataset`
streaming dataset that comes with lit-gpt. You will need to have the tokenizer config available:

```bash
pip install huggingface_hub sentencepiece

python scripts/download.py \
   --repo_id meta-llama/Llama-2-7b-chat-hf \
   --access_token your_hf_token \
   --tokenizer_only true
```

Then, run

```bash
python scripts/prepare_redpajama.py \
  --source_path data/RedPajama-Data-1T \
  --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf/ \
  --destination_path data/lit-redpajama
```

or

```bash
python scripts/prepare_redpajama.py \
  --source_path data/RedPajama-Data-1T-Sample \
  --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf/ \
  --destination_path data/lit-redpajama-sample \
  --sample True
```

for the sample dataset.

In the above we are assuming that you will be using the same tokenizer as used in LLaMA, but any trained [SentencePiece](https://github.com/google/sentencepiece) tokenizer with a 32000 vocabulary size will do here.

The script will take a while to run, so time for :tea: (The 1B sample script takes about 45 min for the data preparation.)

## Pretraining

Running the pretraining script with its default settings requires at least 4 GPUs with 40GB+ each (A100).

```bash
python pretrain/redpajama.py \
  --devices 4 \
  --train_data_dir data/lit-redpajama
```

For running on the sample dataset:

```bash
python pretrain/redpajama.py \
  --devices 4 \
  --train_data_dir data/lit-redpajama-sample
```

The script will save checkpoints periodically to the folder `out/`.

By default, the `pretrain/redpajama.py` script will pretrain the Llama 2 7B model with FSDP in
`bfloat16` precision and gradient accumulation.

You can easily change the size of the model by passing a different string to the model name variable

```python
model_name = "Llama-2-7b-hf"
```

at the top of this script.

The currently supported model names are contained in the [config.py](https://github.com/Lightning-AI/lit-gpt/lit_gpt/config.py) file.
You can

1) either search this file for lines containing "name =",
2) or run `python scripts/download.py` without additional command line arguments

Keep in mind that the original LLaMA training for the 7B model required 83k A100 80GB
hours, so you'll need access to a cluster.

Once you're in a cluster, you can follow [these instructions](https://lightning.ai/docs/fabric/stable/fundamentals/launch.html#launch-on-a-cluster)
to launch the script across machines:

- [SLURM cluster](https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html)
- [Barebones cluster](https://lightning.ai/docs/fabric/stable/guide/multi_node/barebones.html)
- [MPI](https://lightning.ai/docs/fabric/stable/guide/multi_node/other.html)

The [script contains several configurations and hyperparameters](https://github.com/Lightning-AI/lit-gpt/blob/main/pretrain/openwebtext.py#L23-L46) you can tweak.

For instance, `micro_batch_size` should be adjusted so the process will use the available
GPU memory. For more tips to avoid out-of-memory issues, please also see the more detailed
[Dealing with out-of-memory (OOM) errors](oom.md) guide.

Last, logging is kept minimal in the script. In order to use a particular logger
please refer to <https://lightning.ai/docs/fabric/stable/api/loggers.html> or
call a logging client library like `wandb` directly.
