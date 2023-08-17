## Download [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) weights

Vicuna is an open-source family of chatbots trained by fine-tuning LLaMA on user-shared conversations collected from [ShareGPT](https://sharegpt.com).

To see all the available checkpoints for Vicuna, run:

```bash
python scripts/download.py | grep vicuna
```

which will print

```text
lmsys/vicuna-7b-v1.3
lmsys/vicuna-13b-v1.3
lmsys/vicuna-33b-v1.3
lmsys/vicuna-7b-v1.5
lmsys/vicuna-7b-v1.5-16k
lmsys/vicuna-13b-v1.5
lmsys/vicuna-13b-v1.5-16k
```

In order to use a specific Vicuna checkpoint, for instance [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5), download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id lmsys/vicuna-7b-v1.5

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/lmsys/vicuna-7b-v1.5
```

By default, the convert_hf_checkpoint step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install sentencepiece

python chat/base.py --checkpoint_dir checkpoints/lmsys/vicuna-7b-v1.5
```
