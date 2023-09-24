## Download [Code Llama](https://ai.meta.com/blog/code-llama-large-language-model-coding/) weights

Meta developed and publicly released the Code Llama family of large language models (LLMs) on top of Llama 2.

Code Llama models come in three sizes: 7B, 13B, and 34B parameter models. Furthermore, there are three model versions for each size:

- Code Llama: A base model trained on 500B tokens, then and finetuned on 20B tokens.
- Code Llama-Python: The Code Llama model pretrained on 500B tokens, further trained on 100B additional Python code tokens, and then finetuned on 20B tokens.
- Code Llama-Instruct: The Code Llama model trained on 500B tokens, finetuned on 20B tokens, and instruction-finetuned on additional 5B tokens.

All models were  trained on 16,000 token contexts and support generations with up to 100,000 tokens of context.

To see all the available checkpoints, run:

```bash
python scripts/download.py | grep CodeLlama
```

which will print

```text
codellama/CodeLlama-7b-hf
codellama/CodeLlama-7b-Python-hf
codellama/CodeLlama-7b-Instruct-hf
codellama/CodeLlama-13b-hf
codellama/CodeLlama-13b-Python-hf
codellama/CodeLlama-13b-Instruct-hf
codellama/CodeLlama-34b-hf
codellama/CodeLlama-34b-Python-hf
codellama/CodeLlama-34b-Instruct-hf
```

In order to use a specific checkpoint, for instance [CodeLlama-7b-Python-hf](https://huggingface.co/codellama/CodeLlama-7b-Python-hf), download the weights and convert the checkpoint to the lit-gpt format.

```bash
pip install huggingface_hub

python scripts/download.py --repo_id codellama/CodeLlama-7b-Python-hf

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/codellama/CodeLlama-7b-Python-hf
```

By default, the `convert_hf_checkpoint.py` step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install sentencepiece

python chat/base.py --checkpoint_dir checkpoints/codellama/CodeLlama-7b-Python-hf/
```
