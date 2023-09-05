## Download [Llama 2](https://ai.meta.com/llama) weights

Meta developed and publicly released the Llama 2 family of large language models (LLMs), a collection of pretrained and
fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. Its fine-tuned LLMs,
called Llama-2-Chat, are optimized for dialogue use cases. Llama-2-Chat models outperform open-source chat models on
most benchmarks we tested, and in our human evaluations for helpfulness and safety, are on par with some popular
closed-source models like ChatGPT and PaLM.

Llama 2 models are trained on 2 trillion tokens (40% more data than LLaMA 1) and have double the context length of LLaMA 1 (4096 tokens).

Llama 2 comes in a range of parameter sizes — 7B, 13B, and 70B — as well as pretrained and fine-tuned variations.

To see all the available checkpoints, run:

```bash
python scripts/download.py | grep Llama-2
```

which will print

```text
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-13b-hf
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-70b-hf
meta-llama/Llama-2-70b-chat-hf
```

In order to use a specific checkpoint, for instance [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), download the weights and convert the checkpoint to the lit-gpt format.

This requires that you've been granted access to the weights on the HuggingFace hub. You can do so by following the steps at <https://huggingface.co/meta-llama/Llama-2-7b>.
After access is granted, you can find your HF hub token in <https://huggingface.co/settings/tokens>.

```bash
pip install huggingface_hub

python scripts/download.py --repo_id meta-llama/Llama-2-7b-chat-hf --access_token your_hf_token

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf
```

By default, the convert_hf_checkpoint step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install sentencepiece

python chat/base.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf
```
