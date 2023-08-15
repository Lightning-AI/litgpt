## Converting a Lit-GPT Formatted Model to its Original Format

Several models retrieved from external sources must be reformatted with naming conventions for Lit-GPT weights before finetuning. We've provided a helpful script to convert models finetuned with Lit-GPT back to their original format.

The way the script is used depends on the finetuning method. The commands for converting a model back to its original format after using one of the several finetuning methods supported by Lit-GPT are shown below.

> [!NOTE]\
> each example shown below uses Falcon-7B please be sure to update --checkpoint_dir accordingly

### Full Finetuning

After full finetuning, your checkpoint directory will contain a file named `lit_model_finetuned.pth`and converting the finetuned model back to its original weights naming convention can be done by setting the `--checkpoint_name` with:

```sh
python scripts/convert_lit_checkpoint.py \
    --checkpoint_name lit_model_finetuned.pth \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

### Adapter and Adapter V2 Finetuning

After finetuning with either Adapter technique, your checkpoint directory will contain a file named `lit_model_adapter_finetuned.pth` and converting the finetuned model back to its original weights naming convention can be done by setting the `--checkpoint_name` with:

```sh
python scripts/convert_lit_checkpoint.py \
    --checkpoint_name lit_model_adapter_finetuned.pth \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

### LoRA Finetuning

After finetuning with LoRA, your checkpoint directory will contain a file named `lit_model_lora_finetuned.pth` and converting the finetuned model back to its original weights naming convention can be done by setting the `--checkpoint_name` with:

```sh
python scripts/convert_lit_checkpoint.py \
    --checkpoint_name lit_model_lora_finetuned.pth \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b
```
