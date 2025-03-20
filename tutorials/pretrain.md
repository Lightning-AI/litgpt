# Pretrain LLMs with LitGPT


This document explains how to pretrain LLMs using LitGPT.

&nbsp;
## Using the `litgpt pretrain` command

You can pretrain models in LitGPT using the `litgpt pretrain` API starting with any of the available architectures listed by calling `litgpt pretrain list` without any additional arguments:

&nbsp;
> [!TIP]
> To install all required dependencies before pretraining, first run `pip install "litgpt[all]"`.
&nbsp;

```bash
litgpt pretrain list
```

Shown below is an abbreviated list:

```
ValueError: Please specify --model_name <model_name>. Available values:
Camel-Platypus2-13B
...
Gemma-2b
...
Llama-2-7b-hf
...
Mixtral-8x7B-v0.1
...
pythia-14m
```

For demonstration purposes, we can pretrain a small 14 million-parameter Pythia model on the small TinyStories dataset using the [debug.yaml config file](https://github.com/Lightning-AI/litgpt/blob/main/config_hub/pretrain/debug.yaml) as follows:

```bash
litgpt pretrain pythia-14m \
   --config https://raw.githubusercontent.com/Lightning-AI/litgpt/main/config_hub/pretrain/debug.yaml
```


&nbsp;
## Pretrain on custom data

The simplest way to get started with pretraining on a small custom dataset is by using the `TextFiles` data module, which lets you pretrain a dataset from a folder containing plain text files.

&nbsp;

> [!NOTE]
> This approach adds a beginning-of-sequence token at the beginning of each text file. However, it otherwise assumes that you have already cleaned the text files, for example, removing any unwanted characters and inserting beginning-of-sequence and end-of-sequence tokens if applicable in case a text file conists of multiple documents.

&nbsp;

> [!WARNING]
> Using this approach is only recommended for small datasets. Since text data is highly compressible, it is often stored in compressed format, and often in file formats where documents can be loaded row by row without having to load entire files at once. In other words, this `TextFiles` approach is only feasible to store the data in plain text files due to the limited size.
> For datasets that take up multiple gigabytes, we recommend preprocessing it with [LitData](https://github.com/Lightning-AI/litdata) and then reading it from a local directory or S3 connection using `--data LitData`.

&nbsp;

For instance, assume you stored a number of text files in a `custom_pretraining_dataset` folder (we recommend avoiding small files and concatenating them to files of at least 50 Mb for efficiency):

```bash
~ ls -lh custom_pretraining_data
total 3225M
-rw-r--r-- 1 sebastian 50M Apr  2 18:31 combined_1.txt
-rw-r--r-- 1 sebastian 50M Apr  2 18:31 combined_2.txt
-rw-r--r-- 1 sebastian 50M Apr  2 18:31 combined_3.txt
-rw-r--r-- 1 sebastian 50M Apr  2 18:31 combined_4.txt
-rw-r--r-- 1 sebastian 50M Apr  2 18:31 combined_5.txt
...
```

You can then use the `TextFiles` API to pretrain a model (here a small `pythia-14m` model for illustration purposes) from scratch as follows:

```bash
litgpt download EleutherAI/pythia-14m \
  --tokenizer_only true

litgpt pretrain pythia-14m \
   --tokenizer_dir EleutherAI/pythia-14m \
   --data TextFiles \
   --data.train_data_path custom_pretraining_data \
   --train.lr_warmup_steps=200
   --optimizer.lr 0.005
```

&nbsp;
> [!TIP]
> Use the `litgpt pretrain --data.help TextFiles` command to list additional dataset options.
&nbsp;


&nbsp;
## Continued pretraining on custom data

Often, it makes sense to adopt an existing pretrained model and further pretrain it on our own custom data. The existing pretrained model can be either our own pretrained model or a model downloaded from a model hub.

The following subsections illustrate three typical scenarioes:

1. Starting from a downloaded base model
2. Continuing the pretraining after interruption
3. Further pretraining on a different dataset

&nbsp;

> [!NOTE]
> This approach assumes that you have already cleaned the text files, for example, removing any unwanted characters and inserting beginning-of-sequence and end-of-sequence tokens if applicable.

&nbsp;

> [!WARNING]
> Using this approach is only recommended for small datasets. Since text data is highly compressible, it is often stored in compressed format, and often in file formats where documents can be loaded row by row without having to load entire files at once. In other words, this `TextFiles` approach is only feasible to store the data in plain text files due to the limited size.
> For datasets that take up multiple gigabytes, we recommend preprocessing it with [LitData](https://github.com/Lightning-AI/litdata) and then reading it from a local directory or S3 connection using `--data LitData --data.path path/to/your/data`.


&nbsp;
### 1) Continued pretraining when starting from a downloaded base model


For instance, let's assume we download a Pythia model:

```bash
litgpt download EleutherAI/pythia-14m
```

Next, assume we have a custom dataset stored in text files similar to the *Pretrain on custom data* above. We can further pretrain the Pythia model via the `--initial_checkpoint_dir` setting as follows:

```bash
litgpt pretrain pythia-160m \
   --initial_checkpoint_dir EleutherAI/pythia-160m \
   --tokenizer_dir EleutherAI/pythia-160m \
   --out_dir ./new_pretrained_checkpoint \
   --data TextFiles \
   --data.train_data_path custom_pretraining_data \
   --train.max_tokens 1_000_000
```

&nbsp;
> [!TIP]
> Use the `litgpt pretrain --data.help TextFiles` command to list additional dataset options.


&nbsp;
### 2) Continued pretraining after interruption

In case a you interrupted a training run, you can continue it with the `--resume` option, for example:

```bash
litgpt pretrain pythia-160m \
   --resume "auto" \
   --tokenizer_dir EleutherAI/pythia-160m \
   --out_dir ./new_pretrained_checkpoint \
   --data TextFiles \
   --data.train_data_path custom_pretraining_data \
   --train.max_tokens 1_000_000
```

&nbsp;
### 3) Continued pretraining on a new dataset

Suppose you pretrained a model using the examples above. To further pretrain the model on a new dataset, you first need to convert the pretrained checkpoint via the following command:

```bash
litgpt convert_pretrained_checkpoint ./new_pretrained_checkpoint/final ./new_pretrained_checkpoint_converted
```

Then, you can pretrain the converted model on the new dataset as follows:

```bash
litgpt pretrain pythia-160m \
   --initial_checkpoint_dir ./new_pretrained_checkpoint_converted \
   --tokenizer_dir EleutherAI/pythia-160m \
   --out_dir ./new_pretrained_checkpoint_2 \
   --data TextFiles \
   --data.train_data_path custom_pretraining_data_2 \
   --train.max_tokens 1_000_000
```


&nbsp;
## Pretrain a 1.1B TinyLlama model

You can find an end-to-end LitGPT tutorial for pretraining a TinyLlama model using LitGPT [here](pretrain_tinyllama.md).


&nbsp;
## Optimize LitGPT pretraining with Lightning Thunder

[Lightning Thunder](https://github.com/Lightning-AI/lightning-thunder) is a source-to-source compiler for PyTorch, which is fully compatible with LitGPT. In experiments, Thunder resulted in a 40% speed-up compared to using regular PyTorch when finetuning a 7B Llama 2 model.

For more information, see the [Lightning Thunder extension README](https://github.com/Lightning-AI/lightning-thunder).


&nbsp;
## Project templates

The following [Lightning Studio](https://lightning.ai/lightning-ai/studios) templates provide LitGPT pretraining projects in reproducible environments with multi-GPU and multi-node support:
&nbsp;

|                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <p align="left">[Prepare the TinyLlama 1T token dataset](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset) <br> [<img src="https://pl-public-data.s3.amazonaws.com/assets_litgpt/readme/3.webp" width="300"></p>](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset)         | [Pretrain LLMs - TinyLlama 1.1B](https://lightning.ai/lightning-ai/studios/pretrain-llms-tinyllama-1-1b) <br> <p align="left">[<img src="https://pl-public-data.s3.amazonaws.com/assets_litgpt/readme/4.webp" width="300"></p>](https://lightning.ai/lightning-ai/studios/pretrain-llms-tinyllama-1-1b)                                        |
| [Continued Pretraining with TinyLlama 1.1B](https://lightning.ai/lightning-ai/studios/continued-pretraining-with-tinyllama-1-1b) <br> <p align="left">[<img src="https://pl-public-data.s3.amazonaws.com/assets_litgpt/readme/1.webp" width="300"></p>](https://lightning.ai/lightning-ai/studios/continued-pretraining-with-tinyllama-1-1b) | |
|                                                                                                                                                                                                                                                                                                                                             |
