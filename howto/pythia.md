## [Pythia](https://github.com/EleutherAI/pythia) weights

EleutherAI's project Pythia combines interpretability analysis and scaling laws to understand how knowledge develops and evolves during training in autoregressive transformers.
For detailed info on the models, their training, and their behavior, please see its [repository](https://github.com/EleutherAI/pythia).
It includes a suite of 8 checkpoints (weights) on 2 different datasets: the [Pile](https://pile.eleuther.ai/), as well as the Pile with deduplication applied.

The process to use the Pythia weights is the same as described in our [other guides](download_weights.md).

For instance, to run inference with the [pythia-70m](https://huggingface.co/EleutherAI/pythia-70m) checkpoint:

```bash
python scripts/download.py EleutherAI/pythia-70m

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/EleutherAI/pythia-70m

python generate.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/EleutherAI/pythia-70m
```

> **Note**
> The "-deduped" variants of Pythia checkpoints are also supported