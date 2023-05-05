## Downloading pretrained weights

Except for when you are training from scratch, you will need the pretrained weights:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com): git lfs install
git clone stabilityai/stablelm-base-alpha-3b checkpoints/hf-stablelm/stablelm-base-alpha-3b
```

Or if you don't have `git-lfs` installed:

```bash
python scripts/download.py --repo_id stabilityai/stablelm-base-alpha-3b --local_dir checkpoints/hf-stablelm/stablelm-base-alpha-3b
```

Once downloaded, you should have a folder like this:

```text
checkpoints/hf-stablelm
└── stablelm-base-alpha-3b
    ├── ...
    ├── pytorch_model-00001-of-00002.bin
    ├── pytorch_model-00002-of-00002.bin
    ├── pytorch_model.bin.index.json
    └── tokenizer.model
```

Convert the weights to our model format:

```bash
python scripts/convert_hf_checkpoint.py --ckpt_dir checkpoints/hf-stablelm/stablelm-base-alpha-3b --output_dir checkpoints/lit-stablelm/stablelm-base-alpha-3b 
```

> **Note**
> All scripts support argument [customization](customize_paths.md)

Once converted, you should have a folder like this:

```text
checkpoints/lit-stablelm/
├── stablelm-base-alpha-3b
│   ├── config.json
│   └── lit-stablelm.pth
├── tokenizer_config.json
└── tokenizer.json
```

You are all set. Now you can continue with inference or finetuning.

Try running [`generate.py` to test the imported weights](inference.md).
