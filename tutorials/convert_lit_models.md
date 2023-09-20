## Converting Lit-GPT weights to HuggingFace Transformers

HuggingFace Transformer weights need to be converted to a format that Lit-GPT understands with a [conversion script](../scripts/convert_hf_checkpoint.py) before our scripts can run.
We provide a helpful script to convert models Lit-GPT models back to their equivalent HuggingFace Transformers format:

```sh
python scripts/convert_lit_checkpoint.py \
    --checkpoint_path path/to/litgpt/model.pth \
    --output_path where/to/save/the/converted.ckpt \
    --config_path path/to/litgpt/config.json
```

These paths are just placeholders, you will need to customize them based on which finetuning or pretraining script you ran and it's configuration.