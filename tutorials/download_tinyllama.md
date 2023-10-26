## Download TinyLlama weights

[TinyLlama 1.1B](https://github.com/jzhang38/TinyLlama/) is Apache 2.0 licensed and can be used without restrictions.
It is still in development and at the time of writing this, checkpoints for the model trained up to 1T tokens are available.
The target is to train it for ~3 epochs on 3T tokens total. For more details on the schedule and progress of the pretraining, see the official [README](https://github.com/jzhang38/TinyLlama/tree/main).


In order to use the TinyLLama 1.1B model checkpoint, which requires about 5 GB of disk space, download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id PY007/TinyLlama-1.1B-intermediate-step-480k-1T

python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir checkpoints/PY007/TinyLlama-1.1B-intermediate-step-480k-1T
```

You're done! To execute the model just run:

```bash
pip install sentencepiece

python chat/base.py --checkpoint_dir checkpoints/PY007/TinyLlama-1.1B-intermediate-step-480k-1T
```
