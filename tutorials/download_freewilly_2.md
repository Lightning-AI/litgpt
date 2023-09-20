
## Download [FreeWilly 2](https://stability.ai/blog/freewilly-large-instruction-fine-tuned-models) weights

Stability AI announced FreeWilly inspired by the methodology pioneered by Microsoft in its paper: "Orca: Progressive Learning from Complex Explanation Traces of GPT-4”.
FreeWilly2 leverages the Llama 2 70B foundation model to reach a performance that compares favorably with GPT-3.5 for some tasks.

```bash
pip install huggingface_hub

python scripts/download.py --repo_id stabilityai/FreeWilly2

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/stabilityai/FreeWilly2
```

By default, the convert_hf_checkpoint step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install sentencepiece

python chat/base.py --checkpoint_dir checkpoints/stabilityai/FreeWilly2
```
