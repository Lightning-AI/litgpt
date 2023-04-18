## Downloading pretrained weights

Except for when you are training from scratch, you will need the pretrained weights from Meta.
Download the model weights following the instructions on the official [LLaMA repository](https://github.com/facebookresearch/llama).

Once downloaded, you should have a folder like this:

```text
checkpoints/llama
├── 7B
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   └── params.json
├── 13B
│   ...
├── tokenizer_checklist.chk
└── tokenizer.model
```

Convert the weights to the Lit-LLaMA format:

```bash
python scripts/convert_checkpoint.py \
    --output_dir checkpoints/lit-llama \
    --ckpt_dir checkpoints/llama \
    --tokenizer_path checkpoints/llama/tokenizer.model \
    --model_size 7B
```

You are all set. Now you can continue with inference or finetuning.

## Convert from HuggingFace

It is also possible to import weights in the format of the HuggingFace [LLaMA](https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM) model.
Run this script to convert the weights for loading into Lit-LLaMA:

```bash
python scripts/convert_hf_checkpoint.py \
    --hf_checkpoint_path path/to/hf/checkpoint/folder \
    --lit_checkpoint checkpoints/lit-llama.pth
    --model_size 7B
```

You can now run [`generate.py` to test the imported weights](inference.md).