## Download TinyLlama weights

[TinyLlama 1.1B](https://github.com/jzhang38/TinyLlama/) is Apache 2.0 licensed and can be used without restrictions.
It is still in development and at the time of writing this, checkpoints for the model trained up to 2T tokens are available.
The target is to train it for ~3 epochs on 3T tokens total. For more details on the schedule and progress of the pretraining, see the official [README](https://github.com/jzhang38/TinyLlama/tree/main).

There are two version of TinyLlama available: a base one and a fine-tuned "Chat" version.
To see all available version, run:

```bash
python scripts/download.py | grep TinyLlama
```

which will print

```text
TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T
TinyLlama/TinyLlama-1.1B-Chat-v0.6
```

In order to use a specific checkpoint, for instance [TinyLlama 1.1B base model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T), which requires about 5 GB of disk space, download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T

python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T/
```

You're done! To execute the model just run:

```bash
pip install sentencepiece

python chat/base.py --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T/
```
