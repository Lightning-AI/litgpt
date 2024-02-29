## Download [StableCode](https://huggingface.co/collections/stabilityai/stable-code-64f9dfb4ebc8a1be0a3f7650) weights

StableCode is a suite of 4 developer assistant models.

Every one of them is 3 billion parameter decoder-only code completion model pre-trained on diverse set of programming languages that were the top used languages based on the 2023 StackOverflow developer survey.

For more info on the models, please see the [StableCode repository](https://huggingface.co/collections/stabilityai/stable-code-64f9dfb4ebc8a1be0a3f7650).

------

To see all the available checkpoints for StableCode, run:

```bash
python scripts/download.py | grep -E "stable-?code"
```

which will print

```text
stabilityai/stablecode-completion-alpha-3b
stabilityai/stablecode-completion-alpha-3b-4k
stabilityai/stablecode-instruct-alpha-3b
stabilityai/stable-code-3b
```

In order to use a specific StableCode checkpoint, for instance [stable-code-3b](https://huggingface.co/stabilityai/stable-code-3b), download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install 'huggingface_hub[hf_transfer] @ git+https://github.com/huggingface/huggingface_hub'

export repo_id=stabilityai/stable-code-3b
python scripts/download.py --repo_id $repo_id --from_safetensors=True
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$repo_id
```

By default, the `convert_hf_checkpoint` step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install tokenizers

python generate/base.py --prompt "Write in Python softmax function. Be concise." --checkpoint_dir checkpoints/$repo_id
```

Or you can run the model in interactive mode:

```bash
python chat/base.py --checkpoint_dir checkpoints/$repo_id
```
