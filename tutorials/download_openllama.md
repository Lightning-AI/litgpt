## Download [OpenLLaMA](https://github.com/openlm-research/open_llama) weights

OpenLLaMA is a permissively licensed open source reproduction of [Meta AI’s LLaMA](https://github.com/facebookresearch/llama)
7B and 13B checkpoints trained on the [RedPajama dataset](https://github.com/togethercomputer/RedPajama-Data).
The weights can serve as the drop in replacement of LLaMA in existing implementations. We also provide a smaller 3B variant.

To see all the available checkpoints for Open LLaMA, run:

```bash
python scripts/download.py | grep open_llama
```

which will print

```text
openlm-research/open_llama_3b
openlm-research/open_llama_7b
openlm-research/open_llama_13b
```

In order to use a specific OpenLLaMA checkpoint, for instance [open_llama_3b](https://huggingface.co/openlm-research/open_llama_3b), download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id openlm-research/open_llama_3b

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b
```

By default, the convert_hf_checkpoint step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install sentencepiece

python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/openlm-research/open_llama_3b
```
