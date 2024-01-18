## Converting Lit-GPT weights to HuggingFace Transformers

Lit-GPT weights need to be converted to a format that HuggingFace understands with a [conversion script](../scripts/convert_lit_checkpoint.py) before our scripts can run.

We provide a helpful script to convert models Lit-GPT models back to their equivalent HuggingFace Transformers format:

```sh
python scripts/convert_lit_checkpoint.py \
    --checkpoint_path path/to/litgpt/model.pth \
    --output_path where/to/save/the/converted.ckpt \
    --config_path path/to/litgpt/config.json
```

These paths are just placeholders, you will need to customize them based on which finetuning or pretraining script you ran and it's configuration.

Please note that if you want to convert a model that has been fine-tuned using an adapter like LoRA, these weights should be [merged](../scripts/merge_lora.py) to the checkpoint prior to converting.

```sh
python scripts/merge_lora.py \
    --checkpoint_dir path/to/litgpt/model.pth \
    --lora_path path/to/litgpt/lora_finetuned.pth \
    --out_dir where/to/save/the/merged.ckpt
```
