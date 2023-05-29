## Download [StableLM](https://github.com/Stability-AI/StableLM) weights

StableLM is a family of generative language models trained by StabilityAI, trained on a dataset derived from [The Pile](https://pile.eleuther.ai/) but 3x larger, for a total of 1.5 trillion tokens. Weights are released under the [CC-BY-SA license](https://creativecommons.org/licenses/by-sa/4.0).

For more info on the models, please see the [StableLM repository](https://github.com/EleutherAI/pythia). 3B and a 7B checkpoints have been released, both after pre-training and after instruction tuning, using a combination of Stanford's Alpaca, Nomic-AI's gpt4all, RyokoAI's ShareGPT52K datasets, Databricks labs' Dolly, and Anthropic's HH.

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
```

In order to use a specific StableLM checkpoint, for instance [stablelm-base-alpha-3b](http://huggingface.co/stabilityai/stablelm-base-alpha-3b), download the weights and convert the checkpoint to the lit-stablelm format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id stabilityai/stablelm-base-alpha-3b

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

You're done! To execute the model just run:

```bash
python generate.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```
