## Pythia weights

The process to use Pythia weights is the same as described in our [other guides](download_weights.md), just with [modified arguments](customize_paths.md)

For instance, to run inference with the [pythia-70m](https://huggingface.co/EleutherAI/pythia-70m) checkpoint:

```bash
python scripts/download.py --repo_id EleutherAI/pythia-70m --local_dir checkpoints/hf-pythia/pythia-70m

python scripts/convert_hf_checkpoint.py --output_dir checkpoints/lit-pythia/pythia-70m --ckpt_dir checkpoints/hf-pythia/pythia-70m

python generate.py --prompt "Hello, my name is" \
    --checkpoint_path checkpoints/lit-pythia/pythia-70m/lit-stablelm.pth \
    --config_path checkpoints/lit-pythia/pythia-70m/config.json \
    --tokenizer_path checkpoints/lit-pythia/tokenizer.json \
    --tokenizer_config_path checkpoints/lit-pythia/tokenizer_config.json
```

> **Note**
> The "-deduped" variants of Pythia checkpoints are also supported