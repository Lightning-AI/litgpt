# Converting Hugging Face Transformers to LitGPT weights

By default, the `litgpt download` command converts the downloaded HF checkpoint files into a LitGPT compatible format after downloading. For example,

```bash
litgpt download EleutherAI/pythia-14m
```

creates the following files:

```
checkpoints/
└── EleutherAI/
    └── pythia-14m/
        ├── config.json
        ├── generation_config.json
        ├── model_config.yaml      # LitGPT specific file
        ├── lit_model.pth          # LitGPT specific file
        ├── pytorch_model.bin
        ├── tokenizer.json
        └── tokenizer_config.json
```



To disable the automatic conversion, which is useful for development and debugging purposes, you can run the `litgpt download` with the `--convert_checkpoint false` flag. This will only download the checkpoint files but do not convert them for use in LitGPT:

```bash
rm -rf checkpoints/EleutherAI/pythia-14m

litgpt download EleutherAI/pythia-14m \
  --convert_checkpoint false

ls checkpoints/EleutherAI/pythia-14m
```

```
 checkpoints/
└── EleutherAI/
    └── pythia-14m/
        ├── config.json
        ├── generation_config.json
        ├── pytorch_model.bin
        ├── tokenizer.json
        └── tokenizer_config.json
```

The required files `model_config.yaml` and `lit_model.pth` files can then be manually generated via the `litgpt/scripts/convert_hf_checkpoint.py` script:

```bash
litgpt convert_to_litgpt checkpoints/EleutherAI/pythia-14m
```
