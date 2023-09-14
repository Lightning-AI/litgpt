## Converting a Lit-GPT Formatted Model to its Original Format

Several models retrieved from external sources must be reformatted with naming conventions for Lit-GPT weights before finetuning. We've provided a helpful script to convert models finetuned with certain Lit-GPT techniques back to their original format.

The way the script is used depends on the finetuning method. The commands for converting a model back to its original format after using one of the several finetuning methods supported by Lit-GPT are shown below.

> [!NOTE]\
> all examples use the Meta Llama-2-7b-chat-hf checkpoints and Alpaca dataset during finetuning

> [!NOTE]\
> converting Adapter and Adapter V2 finetuned checkpoints cannot be supported until there is a HF model with adapter weights

> [!NOTE]\
> finetuned checkpoints will be saved to a subdirectory in a main directory named `out` e.g. finetuning with LoRA will save checkpoints in `out/lora/alpaca`. that new subdirectory is used as the `--out_dir` argument provided to convert_lit_checkpoint

### Converting Full Finetuned Checkpoints

After full finetuning, your `out_dir` will contain a file named `lit_model_finetuned.pth` and converting the finetuned model back to its original weights naming convention can be done with the following:

```sh
python scripts/convert_lit_checkpoint.py \
    --checkpoint_name lit_model_finetuned.pth  \
    --out_dir out/full/alpaca \
    --model_name Llama-2-7b-chat-hf
```

### Converting LoRA Finetuned Checkpoints

After finetuning with LoRA, your `out_dir` will contain a file named `lit_model_lora_finetuned.pth`. Merging the LoRA weights and converting the finetuned model checkpoint contained in `lit_model_lora_finetuned.pth` can be done with the following:

```sh
python scripts/convert_lit_lora_checkpoint.py \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf \
    --out_dir out/full/alpaca \
    --lora_path out/lora/alpaca/lit_model_lora_finetuned.pth \
    --merge_lora True
```

If you have saved a checkpoint with merged LoRA weights, and only need to convert that checkpoint, you can do:

```sh
python scripts/convert_lit_lora_checkpoint.py \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf \
    --out_dir out/full/alpaca
```

A
