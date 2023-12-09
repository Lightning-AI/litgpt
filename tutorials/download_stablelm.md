## Download [StableLM](https://github.com/Stability-AI/StableLM) weights

StableLM is a family of generative language models trained by StabilityAI, trained on a dataset derived from [The Pile](https://pile.eleuther.ai/) but 3x larger, for a total of 1.5 trillion tokens. Weights are released under the [CC-BY-SA license](https://creativecommons.org/licenses/by-sa/4.0).

For more info on the models, please see the [StableLM repository](https://github.com/Stability-AI/StableLM). 3B and a 7B checkpoints have been released, both after pre-training and after instruction tuning, using a combination of Stanford's Alpaca, Nomic-AI's gpt4all, RyokoAI's ShareGPT52K datasets, Databricks labs' Dolly, and Anthropic's HH.

To see all the available checkpoints for StableLM, run:

```bash
python scripts/download.py | grep stablelm
```

which will print

```text
stabilityai/stablelm-base-alpha-3b
stabilityai/stablelm-base-alpha-7b
stabilityai/stablelm-tuned-alpha-3b
stabilityai/stablelm-tuned-alpha-7b
stabilityai/stablelm-zephyr-3b
```

In order to use a specific StableLM checkpoint, for instance [stablelm-base-alpha-3b](http://huggingface.co/stabilityai/stablelm-base-alpha-3b), download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id stabilityai/stablelm-base-alpha-3b

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

By default, the convert_hf_checkpoint step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install tokenizers

python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

------

### StableLM Zephyr 3B

Lightweight LLM, preference tuned for instruction following and Q&A-type tasks. This model is an extension of the pre-existing StableLM 3B-4e1t model and is inspired by the Zephyr 7B model from HuggingFace. With StableLM Zephyr's 3 billion parameters, this model efficiently caters to a wide range of text generation needs, from simple queries to complex instructional contexts on edge devices.
More details can be found in the [announcement](https://stability.ai/news/stablelm-zephyr-3b-stability-llm).

In order to use this model, download the weights and convert the checkpoint to the lit-gpt format. As this version of the model is in `safetensor` format, to download it an additional flag is required:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id stabilityai/stablelm-zephyr-3b --from_safetensors=True
```

The model is shipped in `bfloat16` format, so if your hardware doesn't support it, you can provide `dtype` argument during model conversion. For example we can convert the weights into `float32` format:

```bash
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/stabilityai/stablelm-zephyr-3b --dtype float32
```

You're done! To start a chat with this model just run:

```bash
python chat/base.py --checkpoint_dir checkpoints/stabilityai/stablelm-zephyr-3b
```
