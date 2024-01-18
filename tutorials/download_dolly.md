## Download [Dolly](https://github.com/databrickslabs/dolly) weights

Databricks’ [Dolly](https://huggingface.co/databricks/dolly-v2-12b) is an instruction-following large language model trained on the Databricks machine learning platform
that is licensed for commercial use. Based on `pythia-12b`, Dolly is trained on ~15k instruction/response fine tuning records
[`databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) generated
by Databricks employees in capability domains from the InstructGPT paper, including brainstorming, classification, closed QA, generation,
information extraction, open QA and summarization. `dolly-v2-12b` is not a state-of-the-art model, but does exhibit surprisingly
high quality instruction following behavior not characteristic of the foundation model on which it is based.

For detailed info on the models, their training, and their behavior, please see the [Dolly repository](https://github.com/databrickslabs/dolly).

To see all the available checkpoints for Dolly, run:

```bash
python scripts/download.py | grep dolly
```

which will print

```text
databricks/dolly-v2-3b
databricks/dolly-v2-7b
databricks/dolly-v2-12b
```

In order to use a specific Dolly checkpoint, for instance [dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b), download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id databricks/dolly-v2-3b

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/databricks/dolly-v2-3b
```

By default, the convert_hf_checkpoint step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install tokenizers

python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/databricks/dolly-v2-3b
```
