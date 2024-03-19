# Finetuning

We provide a simple training scripts (`litgpt/finetune/*.py`) that instruction-tunes a pretrained model on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.
For example, you can either use

LoRA ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)):

```bash
litgpt finetune lora
```

or Adapter ([Zhang et al. 2023](https://arxiv.org/abs/2303.16199)):

```bash
litgpt finetune adapter
```

or Adapter v2 ([Gao et al. 2023](https://arxiv.org/abs/2304.15010)):

```bash
litgpt finetune adapter_v2
```


The finetuning requires at least one GPU with ~12 GB memory (RTX 3060).

It is expected that you have downloaded the pretrained weights as described above.
More details about each finetuning method and how you can apply it to your own data can be found in our technical how-to guides.


### Finetuning how-to guides

These technical tutorials illustrate how to run the finetuning code.

- [Full-parameter finetuning](finetune_full.md)
- [Finetune with Adapters](finetune_adapter.md)
- [Finetune with LoRA or QLoRA](finetune_lora.md)

&nbsp;

### Understanding finetuning -- conceptual tutorials

Looking for conceptual tutorials and explanations? We have some additional articles below:

- [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)

- [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)
