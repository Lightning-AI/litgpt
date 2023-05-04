# Pre-train LLaMA on RedPajama

This howto will walk you through setting up the RedPajama dataset and launching the pre-training script.

## What's RedPajama

[RedPajama](https://github.com/togethercomputer/RedPajama-Data) is an open-source reproduction of the original LLaMA training dataset.

It contains a total of 1.2 trillion tokens, divided into

```text
Commoncrawl   878B
C4            175B
GitHub         59B
Books          26B
ArXiv          28B
Wikipedia      24B
StackExchange  20B
```

The [RedPajama repo](https://github.com/togethercomputer/RedPajama-Data) contains the source code for collecting and preparing
the dataset, and it is Apache 2.0 licensed.

The data itself is licensed according to the original licenses with which its invidivdual parts were released.
The GitHub datasets are limited to MIT, BSD, or Apache 2.0 repositories.

Along with the full [RedPajama-1T dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T),
the [RedPajama-1T-Sample](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) 1B sample dataset
is also available for development.

You can download the data using git lfs:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com): git lfs install
git clone https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T data/RedPajama-Data-1T
```

```bash
# Make sure you have git-lfs installed (https://git-lfs.com): git lfs install
git clone https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample data/RedPajama-Data-1T-Sample
```

## Prepare RedPajama for training

The dataset consists of 2084 `jsonl` files (the sample dataset contains 11). In order to start pre-training lit-llama
on it, you need to read, tokenize, and write the data in binary chunks. This will leverage the `PackedDataset`
streaming dataset that comes with lit-llama.

Do to so, run

```bash
python scripts/prepare_redpajama.py --source_path data/RedPajama-Data-1T --tokenizer_path checkpoints/lit-llama/tokenizer.model --destination_path data/lit-redpajama
```

or

```bash
python scripts/prepare_redpajama.py --source_path data/RedPajama-Data-1T-Sample --tokenizer_path checkpoints/lit-llama/tokenizer.model --destination_path data/lit-redpajama-sample --sample True
```

for the sample dataset.

In the above we are assuming that you will be using the same tokenizer as used in LLaMA, but any trained [SentencePiece](https://github.com/google/sentencepiece) tokenizer with a 32000 vocabulary size will do here.

The script will take a while to run, so time for :tea:

## Pre-training

Running the pre-training script requires at least 4 GPUs with 40GB+ each (A100).

```bash
python train_redpajama.py --devices 4 --train_data_dir data/lit-redpajama
```

For running on the sample dataset:

```bash
python train_redpajama.py --devices 4 --train_data_dir data/lit-redpajama-sample
```

The script will save checkpoints periodically to the folder `out/`.

The `train_redpajama.py` script will pre-train the LLaMA 7B model with FSDP in
`bfloat16` precision and gradient accumulation.

You can easily change the size of the model by passing a different string to

```python
config = LLaMAConfig.from_name("7B")
```

in the `main` function.

Keep in mind that the original LLaMA training for the 7B model required 83k A100 80GB
hours, so you'll need access to a cluster.

Once you're in a cluster, you can follow [these instructions](https://lightning.ai/docs/fabric/stable/guide/multi_node/other.html)
to launch the script across machines:

- [SLURM cluster](https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html)
- [Barebones cluster](https://lightning.ai/docs/fabric/stable/guide/multi_node/barebones.html)
- [MPI](https://lightning.ai/docs/fabric/stable/guide/multi_node/other.html)

The script contains several configurations and hyperparameters you can tweak:

```python
out_dir = "out/training"
save_interval = 1000
eval_interval = 1000
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 6e-4
batch_size = 125
micro_batch_size = 5
max_iters = 600000  # num_epochs * epoch_size // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5
```

In particular, `micro_batch_size` should be adjusted so the process will use the available
GPU memory.

Last, logging is kept minimal in the script. In order to use a particular logger
please refer to <https://lightning.ai/docs/fabric/stable/api/loggers.html> or
call a logging client library like `wandb` directly.
