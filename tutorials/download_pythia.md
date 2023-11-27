## Download [Pythia](https://github.com/EleutherAI/pythia) weights

EleutherAI's project Pythia combines interpretability analysis and scaling laws to understand how knowledge develops and evolves during training in autoregressive transformers. Weights are released under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

For detailed info on the models, their training, and their behavior, please see the [Pythia repository](https://github.com/EleutherAI/pythia).
It includes a suite of 8 checkpoints (weights) on 2 different datasets: [The Pile](https://pile.eleuther.ai/), as well as The Pile with deduplication applied. In addition there are two small models that come only in non-deduplicated form: `Pythia-14m` and `Pythia-31m`.

To see all the available checkpoints for Pythia, run:

```bash
python scripts/download.py | grep pythia
```

which will print

```text
EleutherAI/pythia-14m
EleutherAI/pythia-31m
EleutherAI/pythia-70m
EleutherAI/pythia-160m
EleutherAI/pythia-410m
EleutherAI/pythia-1b
EleutherAI/pythia-1.4b
EleutherAI/pythia-2.8b
EleutherAI/pythia-6.9b
EleutherAI/pythia-12b
EleutherAI/pythia-70m-deduped
EleutherAI/pythia-160m-deduped
EleutherAI/pythia-410m-deduped
EleutherAI/pythia-1b-deduped
EleutherAI/pythia-1.4b-deduped
EleutherAI/pythia-2.8b-deduped
EleutherAI/pythia-6.9b-deduped
EleutherAI/pythia-12b-deduped
```

In order to use a specific Pythia checkpoint, for instance [pythia-1b](https://huggingface.co/EleutherAI/pythia-1b), download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id EleutherAI/pythia-1b

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/EleutherAI/pythia-1b
```

By default, the convert_hf_checkpoint step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install tokenizers

python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/EleutherAI/pythia-1b
```
