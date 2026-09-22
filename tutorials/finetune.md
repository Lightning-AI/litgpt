# Finetuning

We provide a simple finetuning commands (`litgpt finetune_*`) that instruction-finetune a pretrained model on datasets such as [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm), and others. For more information on the supported instruction datasets and how to prepare your own custom datasets, please see the [tutorials/prepare_dataset](prepare_dataset.md) tutorials.

LitGPT currently supports the following finetuning methods:

```bash
litgpt finetune_full
litgpt finetune_lora
litgpt finetune_adapter
litgpt finetune_adapter_v2
```

&nbsp;
> [!TIP]
> To install all required dependencies before finetuning, first run `pip install "litgpt[all]"`.
&nbsp;


The following sections explain tokenizer considerations when finetuning on a new language and provide more details about these methods, including links for additional resources.


&nbsp;
## Finetuning on a new language

### Do I need to add tokens?

Not necessarily. A word or character does not need its own vocabulary entry to be represented by a tokenizer. For example, [Falcon-7B's tokenizer](https://huggingface.co/tiiuae/falcon-7b/blob/main/tokenizer.json) uses byte-level BPE, so Chinese text can be represented using existing tokens even when a complete Chinese character is not a single token. Finetuning uses the checkpoint's tokenizer; it does not learn a new vocabulary or automatically add tokens from the training dataset.

Start with the original tokenizer and check representative samples from your dataset:

- Encode and decode complete samples with `litgpt.Tokenizer`, without adding beginning-of-sequence or end-of-sequence tokens (`bos=False, eos=False`), and inspect whether the text is preserved. Check for unexpected normalization or unknown tokens when using other tokenizers.
- Measure the tokenized sequence lengths. A language may require multiple tokens per character, leaving less text within the training sequence length and increasing truncation.
- Evaluate the finetuned model on held-out examples in the target language. Being able to encode the text does not mean the pretrained model already understands that language well.

Use the [custom dataset preparation guide](prepare_dataset.md) to prepare your data for finetuning.

### Why doesn't the embedding parameter count increase?

The number of embedding parameters depends on the model's vocabulary capacity and embedding dimension, not on the number of distinct words or characters in your dataset. With the original tokenizer and model configuration, this count stays unchanged. Full finetuning updates existing embedding weights without adding parameters. LitGPT's standard LoRA finetuning freezes the original token embeddings and trains the selected low-rank adapters instead; the reported trainable parameter count is not a count of newly encountered tokens.

### What if I intentionally extend the vocabulary?

Adding tokens is a separate model adaptation task, not a prerequisite for using Chinese data with Falcon-7B. LitGPT's finetuning commands do not automatically resize a checkpoint when its tokenizer changes. Editing only `tokenizer.json` or `model_config.yaml` is insufficient:

- Preserve all existing token IDs so that they still refer to the pretrained embedding rows.
- Ensure that both the input embedding (`transformer.wte`) and output projection (`lm_head`) cover every token ID. LitGPT uses `padded_vocab_size` for these dimensions; newly added IDs may fit within existing padding, but those rows have not been trained to represent the new tokens.
- If expansion is needed, preserve existing weights and initialize the new rows in both matrices. Keep the saved model configuration and checkpoint tensor shapes consistent; changing the configuration alone causes shape mismatches when loading the original weights.
- Make the input and output rows for the added tokens trainable and include them in the optimizer and saved checkpoint. The standard LoRA workflow is not sufficient for learning and saving newly initialized input embeddings because it freezes those embeddings and saves adapter weights.
- Save the matching tokenizer with the adapted checkpoint and use it consistently for dataset preparation, training, and inference. Retokenize any previously prepared token-ID datasets after changing the tokenizer.

Vocabulary expansion therefore needs a custom training and checkpoint-saving workflow. Keep the original vocabulary unless you have measured a need for this additional work.


&nbsp;
## LitGPT finetuning commands

The section below provides additional information on the available and links to further resources.

&nbsp;
### Full finetuning

```bash
litgpt finetune_full
```

This method trains all model weight parameters and is the most memory-intensive finetuning technique in LitGPT.

**More information and resources:**

- the LitGPT [tutorials/finetune_full](finetune_full.md) tutorial


&nbsp;
### LoRA and QLoRA finetuning

```bash
litgpt finetune_lora stabilityai/stablelm-base-alpha-3b
```

LoRA and QLoRA are parameter-efficient finetuning technique that only require updating a small number of parameters, which makes this a more memory-efficienty alternative to full finetuning.

**More information and resources:**

- the LitGPT [tutorials/finetune_lora](finetune_lora.md) tutorial
- the LoRA paper by ([Hu et al. 2021](https://arxiv.org/abs/2106.09685))
- the conceptual tutorial [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)


&nbsp;
### Adapter finetuning

```bash
litgpt finetune_adapter stabilityai/stablelm-base-alpha-3b
```

or

```bash
litgpt finetune_adapter_v2 stabilityai/stablelm-base-alpha-3b
```

Similar to LoRA, adapter finetuning is a parameter-efficient finetuning technique that only requires training a small subset of weight parameters, making this finetuning method more memory-efficient than full-parameter finetuning.

**More information and resources:**

- the LitGPT [tutorials/finetune_adapter](finetune_adapter.md) tutorial
- the Llama-Adapter ([Gao et al. 2023](https://arxiv.org/abs/2304.15010)) and Llama-Adapter v2  ([Zhang et al. 2023](https://arxiv.org/abs/2303.16199)) papers that originally introduces these methods
- the conceptual tutorial [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)
