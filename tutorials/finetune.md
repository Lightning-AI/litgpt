# Finetuning

We provide a simple finetuning commands (`litgpt finetune *`) that instruction-finetune a pretrained model on datasets such as [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm), and others. For more information on the supported instruction datasets and how to prepare your own custom datasets, please see the [tutorials/prepare_dataset](prepare_dataset.md) tutorials.

LitGPT currently supports the following finetuning methods:

```bash
litgpt finetune full
litgpt finetune lora
litgpt finetune qlora
litgpt finetune adapter
litgpt finetune adapter_v2
```

The following section provides more details about these methods, including links for additional resources.


&nbsp;
## LitGPT finetuning commands

The section below provides additional information on the available and links to further resources.

&nbsp;
### Full finetuning

```bash
litgpt finetune full
```

This method trains all model weight parameters and is the most memory-intensive finetuning technique in LitGPT.

**More information and resources:**

- the LitGPT [tutorials/finetune_full](finetune_full.md) tutorial


&nbsp;
### LoRA and QLoRA finetuning

```bash
litgpt finetune lora
```

```bash
litgpt finetune qlora
```


LoRA and QLoRA (short for quantized LoRA) are parameter-efficient finetuning technique that only require updating a small number of parameters, which makes this a more memory-efficienty alternative to full finetuning.

**More information and resources:**

- the LitGPT [tutorials/finetune_lora](finetune_lora.md) tutorial
- the LoRA paper by ([Hu et al. 2021](https://arxiv.org/abs/2106.09685))
- the conceptual tutorial [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)


&nbsp;
### Adapter finetuning

```bash
litgpt finetune adapter
```

or

```bash
litgpt finetune adapter_v2
```

Simillar to LoRA, adapter finetuning is a parameter-efficient finetuning technique that only requires training a small subset of weight parameters, making this finetuning method more memory-efficient than full-parameter finetuning. 

**More information and resources:**

- the LitGPT [tutorials/finetune_adapter](finetune_adapter.md) tutorial
- the Llama-Adapter ([Gao et al. 2023](https://arxiv.org/abs/2304.15010)) and Llama-Adapter v2  ([Zhang et al. 2023](https://arxiv.org/abs/2303.16199)) papers that originally introduces these methods
- the conceptual tutorial [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)
