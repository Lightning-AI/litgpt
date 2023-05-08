## Download [StableLM](https://github.com/Stability-AI/StableLM) weights

Except for when you are training from scratch, you will need the pretrained weights:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com): git lfs install
git clone https://huggingface.co/stabilityai/stablelm-base-alpha-3b checkpoints/stabilityai/stablelm-base-alpha-3b
```

Or if you don't have `git-lfs` installed:

```bash
pip install huggingface_hub
python scripts/download.py stabilityai/stablelm-base-alpha-3b
```

Once downloaded, you should have a folder like this:

```text
checkpoints/stabilityai
└── stablelm-base-alpha-3b
    ├── ...
    ├── pytorch_model-00001-of-00002.bin
    ├── pytorch_model-00002-of-00002.bin
    ├── pytorch_model.bin.index.json
    └── tokenizer.model
```

Convert the weights to our model format:

```bash
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

Once converted, you should have two added files:

```text
checkpoints/stabilityai
└── stablelm-base-alpha-3b
    ├── ...
    ├── lit_config.json
    └── lit_model.pth
```

You are all set. Now you can continue with inference or finetuning.

Try running [`generate.py` to test the imported weights](inference.md).
