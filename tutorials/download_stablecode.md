## Download [StableCode](https://huggingface.co/collections/stabilityai/stable-code-64f9dfb4ebc8a1be0a3f7650) weights

StableCode is a suite of 4 developer assistant models.

Each one of them is a decoder-only code completion model with 3 billion parameters, pre-trained on a diverse collection of programming languages that ranked highest in the 2023 StackOverflow developer survey.

For more info on the models, please visit the [StableCode repository](https://huggingface.co/collections/stabilityai/stable-code-64f9dfb4ebc8a1be0a3f7650).

------

To see all the available checkpoints for StableCode, run:

```bash
python litgpt/scripts/download.py | grep -E "stable-?code"
```

which will print:

```text
stabilityai/stablecode-completion-alpha-3b
stabilityai/stablecode-completion-alpha-3b-4k
stabilityai/stablecode-instruct-alpha-3b
stabilityai/stable-code-3b
```

In order to use a specific StableCode checkpoint, for instance [stable-code-3b](https://huggingface.co/stabilityai/stable-code-3b), download the weights and convert the checkpoint to the LitGPT format:

```bash
pip install 'huggingface_hub[hf_transfer] @ git+https://github.com/huggingface/huggingface_hub'

export repo_id=stabilityai/stable-code-3b
python litgpt/scripts/download.py --repo_id $repo_id
```

By default, the checkpoint conversion step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install tokenizers

python litgpt/generate/base.py --prompt "Write in Python a softmax function. Be concise." --checkpoint_dir checkpoints/$repo_id
```

Or you can run the model in an interactive mode:

```bash
python litgpt/chat/base.py --checkpoint_dir checkpoints/$repo_id
```
