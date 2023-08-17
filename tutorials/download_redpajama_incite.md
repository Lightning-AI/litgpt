## Download [RedPajama-INCITE](https://www.together.xyz/blog/redpajama-models-v1) weights

Togethercomputer's RedPajama-INCITE family of models were trained over the [RedPajama v1](https://www.together.xyz/blog/redpajama) dataset, with the same architecture as the popular [Pythia](download_pythia.md) model suite. Weights are released under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

The release includes a base model, a chat fine-tuned model, and an instruction tuned model of sizes 3B and 7B.

To see all the available checkpoints for RedPajama-INCITE, run:

```bash
python scripts/download.py | grep RedPajama
```

which will print

```text
togethercomputer/RedPajama-INCITE-Base-3B-v1
togethercomputer/RedPajama-INCITE-Chat-3B-v1
togethercomputer/RedPajama-INCITE-Instruct-3B-v1
togethercomputer/RedPajama-INCITE-7B-Base
togethercomputer/RedPajama-INCITE-7B-Chat
togethercomputer/RedPajama-INCITE-7B-Instruct
togethercomputer/RedPajama-INCITE-Base-7B-v0.1
togethercomputer/RedPajama-INCITE-Chat-7B-v0.1
togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1
```

In order to use a specific RedPajama-INCITE checkpoint, for instance [RedPajama-INCITE-Base-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1), download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id togethercomputer/RedPajama-INCITE-Base-3B-v1

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/togethercomputer/RedPajama-INCITE-Base-3B-v1
```

By default, the convert_hf_checkpoint step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install tokenizers

python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/togethercomputer/RedPajama-INCITE-Base-3B-v1
```
