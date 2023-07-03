## Download [LongChat](https://lmsys.org/blog/2023-06-29-longchat) weights

LongChat is an open-source family of chatbots based on LLaMA featuring an extended context length up to 16K tokens.
The technique used to extend the context length is described in [this blogpost](https://kaiokendev.github.io/context).

To see all the available checkpoints, run:

```bash
python scripts/download.py | grep longchat
```

which will print

```text
lmsys/longchat-7b-16k
lmsys/longchat-13b-16k
```

In order to use a specific checkpoint, for instance [longchat-7b-16k](https://huggingface.co/lmsys/longchat-7b-16k), download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id lmsys/longchat-7b-16k

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/lmsys/longchat-7b-16k
```

You're done! To execute the model just run:

```bash
pip install sentencepiece

python chat/base.py --checkpoint_dir checkpoints/lmsys/longchat-7b-16k
```
