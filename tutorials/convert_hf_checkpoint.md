# Converting Hugging Face Transformers to Lit-GPT weights

By default, the `scripts/download.py` script converts the downloaded HF checkpoint files into a Lit-GPT compatible format after downloading. For example,

```bash
python scripts/download.py --repo_id EleutherAI/pythia-14m
```

creates the following files:

```
checkpoints/
└── EleutherAI/
    └── pythia-14m/
        ├── config.json
        ├── generation_config.json
        ├── lit_config.json        # Lit-GPT specific file
        ├── lit_model.pth          # Lit-GPT specific file
        ├── pytorch_model.bin
        ├── tokenizer.json
        └── tokenizer_config.json
```



To disable the automatic conversion, which is useful for development and debugging purposes, you can run the `scripts/download.py` with the `--convert_checkpoint false` flag. This will only download the checkpoint files but do not convert them for use in Lit-GPT:

```bash
rm -rf checkpoints/EleutherAI/pythia-14m 

python scripts/download.py \
  --repo_id EleutherAI/pythia-14m \
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

The required files `lit_config.json` and `lit_model.pth` files can then be manually generated via the `scripts/convert_hf_checkpoint.py` script:

```bash
python scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/EleutherAI/pythia-14m
```

