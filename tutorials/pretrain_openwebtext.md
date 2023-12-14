# Pretrain Llama 2 on OpenWebText

This tutorial will walk you through setting up the OpenWebText dataset and launching the pretraining script.

## What's OpenWebText

[OpenWebText](https://github.com/jcpeterson/openwebtext) is an open-source reproduction of OpenAI's unreleased WebText training dataset, which was originally used to train GPT-2. The version that is used here consists of 8M documents and is loaded via the `load_dataset("openwebtext", ...)` function from the [datasets](https://github.com/huggingface/datasets) Python package. [Please refer to the website hosting the dataset](https://huggingface.co/datasets/Skylion007/openwebtext) for the licensing information.

## Prepare OpenWebText for training

In order to start pretraining lit-gpt on it, you need to read, tokenize, and write the data in binary format.

To prepare the dataset with the Llama 2 tokenizer, run

```bash
pip install datasets

python scripts/prepare_openwebtext.py \
  --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf/ \
  --destination_path data/openwebtext
```

The script will take about 15 min to run.

## Pretraining

Running the pretraining script with its default settings requires at least 4 GPUs with 40GB+ each. (However, alternatively, you can train a smaller Pythia-70m on 1 GPU, more information about that further below).

```bash
python pretrain/openwebtext.py \
  --devices 4
```

The script will save checkpoints periodically to the folder `out/`.

By default, the `pretrain/openwebtext.py` script will pretrain the Llama 2 7B model with FSDP in
`bfloat16` precision and gradient accumulation.

You can easily change the size of the model by passing a different string to the model name variable

```python
model_name = "Llama-2-7b-hf"
```

at the top of this script.

The currently supported model names are contained in the [config.py](https://github.com/Lightning-AI/lit-gpt/lit_gpt/config.py) file.
You can

1) either search this file for lines containing "name =",
2) or run `python scripts/download.py` without additional command line arguments,

Keep in mind that the original LLaMA training for the 7B model required 83k A100 80GB
hours (on a bigger dataset). However, for full pretraining on OpenWebText, you'll likely still need access to a cluster.

Once you're in a cluster, you can follow [these instructions](https://lightning.ai/docs/fabric/stable/fundamentals/launch.html#launch-on-a-cluster)
to launch the script across machines:

- [SLURM cluster](https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html)
- [Barebones cluster](https://lightning.ai/docs/fabric/stable/guide/multi_node/barebones.html)
- [MPI](https://lightning.ai/docs/fabric/stable/guide/multi_node/other.html)

The [script contains several configurations and hyperparameters](https://github.com/Lightning-AI/lit-gpt/blob/main/pretrain/redpajama.py#L23-L45) you can tweak.

For instance, `micro_batch_size` should be adjusted so the process will use the available
GPU memory. For more tips to avoid out-of-memory issues, please also see the more detailed
[Dealing with out-of-memory (OOM) errors](oom.md) guide.

Last, logging is kept minimal in the script. In order to use a particular logger
please refer to <https://lightning.ai/docs/fabric/stable/api/loggers.html> or
call a logging client library like `wandb` directly.

## Training a smaller model on a single GPU

To train a smaller Pythia 70M model on a single GPU, you can modify the `pretrain/openwebtext.py` file to use the following settings:

```python
model_name = "pythia-70m"
```

(Please see the the `download_*` scripts in the [tutorials](.) for more information on downloading model checkpoints for different models.)

Also, before you start training, note that you will need to prepare the dataset specifically for this model since it may use a different tokenizer:

```bash
python scripts/prepare_openwebtext.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-70m/ \
  --destination_path data/lit-openwebtext

python pretrain/openwebtext.py \
  --devices 4
```

## Using the PyTorch Lightning `Trainer`

The `pretrain/openwebtext.py` used and discussed above uses Lightning Fabric, which is an open-source library for accessing more advanced PyTorch features conveniently (for example, mixed-precision training, multi-GPU training like FSDP, and more).

The PyTorch Lightning Trainer, which shares the same accelerator code with Fabric, offers additional features, such as more advanced checkpointing and logging. If you prefer using the PyTorch Lightning Trainer, you can use the alternative `pretrain/openwebtext_trainer.py` script:

```bash
python pretrain/openwebtext_trainer.py \
  --devices 4
```
