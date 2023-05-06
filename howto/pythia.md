## Pythia weights

The process to use Pythia weights is the same as described in our [other guides](download_weights.md).

For instance, to run inference with the [pythia-70m](https://huggingface.co/EleutherAI/pythia-70m) checkpoint:

```bash
python scripts/download.py EleutherAI/pythia-70m

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/EleutherAI/pythia-70m

python generate.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/EleutherAI/pythia-70m
```

> **Note**
> The "-deduped" variants of Pythia checkpoints are also supported