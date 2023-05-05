## Downloading pretrained weights

Except for when you are training from scratch, you will need the pretrained weights from Meta.

### Original Meta weights

Download the model weights following the instructions on the official [LLaMA repository](https://github.com/facebookresearch/llama).

Once downloaded, you should have a folder like this:

```text
checkpoints/llama
├── 7B
│   ├── ...
│   └── consolidated.00.pth
├── 13B
│   ...
└── tokenizer.model
```

Convert the weights to the Lit-LLaMA format:

```bash
python scripts/convert_checkpoint.py --model_size 7B
```

> **Note**
> All scripts support argument [customization](customize_paths.md)

### OpenLLaMA

OpenLM Research has released **Apache 2.0 licensed** weights obtained by training LLaMA on the 1.2 trillion token open-source [RedPajama](https://github.com/togethercomputer/RedPajama-Data) dataset.

Weights were released in preview on intermediate number of tokens (200B, 300B at the time of writing). In order to get them do:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com): git lfs install
git clone https://huggingface.co/openlm-research/open_llama_7b_preview_300bt checkpoints/open-llama/7B
```

Or if you don't have `git-lfs` installed:

```bash
python scripts/download.py --repo_id openlm-research/open_llama_7b_preview_300bt --local_dir checkpoints/open-llama/7B
```

Once downloaded, you should have a folder like this:

```text
checkpoints/open-llama/
└── 7B
    └── open_llama_7b_preview_300bt_transformers_weights
        ├── ...
        ├── pytorch_model-00001-of-00002.bin
        ├── pytorch_model-00002-of-00002.bin
        ├── pytorch_model.bin.index.json
        └── tokenizer.model
```

Convert the weights to the Lit-LLaMA format:

```bash
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/open-llama/7B/open_llama_7b_preview_300bt_transformers_weights --model_size 7B
```

> **Note**
> All scripts support argument [customization](customize_paths.md)

Once converted, you should have a folder like this:

```text
checkpoints/lit-llama/
├── 7B
│   └── lit-llama.pth
└── tokenizer.model
```

You are all set. Now you can continue with inference or finetuning.

Try running [`generate.py` to test the imported weights](inference.md).


### Alternative sources

You might find LLaMA weights hosted online in the HuggingFace hub. Beware that this infringes the original weight's license.
You could try downloading them by running the following command with a specific repo id:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com): git lfs install
git clone REPO_ID checkpoints/hf-llama/7B
```

Or if you don't have `git-lfs` installed:

```bash
python scripts/download.py --repo_id REPO_ID --local_dir checkpoints/hf-llama/7B
```

Once downloaded, you should have a folder like this:

```text
checkpoints/hf-llama/
└── 7B
    ├── ...
    ├── pytorch_model-00001-of-00002.bin
    ├── pytorch_model-00002-of-00002.bin
    ├── pytorch_model.bin.index.json
    └── tokenizer.model
```

Convert the weights to the Lit-LLaMA format:

```bash
python scripts/convert_hf_checkpoint.py --model_size 7B
```

> **Note**
> All scripts support argument [customization](customize_paths.md)

Once converted, you should have a folder like this:

```text
checkpoints/lit-llama/
├── 7B
│   └── lit-llama.pth
└── tokenizer.model
```

You are all set. Now you can continue with inference or finetuning.

Try running [`generate.py` to test the imported weights](inference.md).
