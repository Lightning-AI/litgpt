## Download TinyLlama weights

[TinyLlama 1.1B](https://github.com/jzhang38/TinyLlama/) is Apache 2.0 licensed and can be used without restrictions.
It is still in development and at the time of writing this, checkpoints for the model trained up to 2T tokens are available.
The target is to train it for ~3 epochs on 3T tokens total. For more details on the schedule and progress of the pretraining, see the official [README](https://github.com/jzhang38/TinyLlama/tree/main).

There are two version of TinyLlama available: a base one and a fine-tuned "Chat" version.
To see all available versions, run:

```bash
python scripts/download.py | grep TinyLlama
```

which will print

```text
TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

In order to use a specific checkpoint, for instance [TinyLlama 1.1B base model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T), which requires about 5 GB of disk space, download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

-----

With the `Chat` version of the model, the download and conversion procedures are slightly different.
As this version of the model is stored in `safetensor` format, to download it an additional flag is required:

```bash
python scripts/download.py --repo_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --from_safetensors=True
```

The model is shipped in `bfloat16` format, so if your hardware doesn't support it, you can provide `dtype` argument during model conversion. For example we can convert the weights into `float32` format:

```bash
python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype=float32
```

-----

You're done! To execute the model just run:

```bash
pip install sentencepiece

# base version
python chat/base.py --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

or

# chat version
python chat/base.py --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

To improve the response from Chat version you can also provide these args (as in the [model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)):

```bash
python chat/base.py --checkpoint_dir checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0 --top_k=50 --temperature=0.7
```
