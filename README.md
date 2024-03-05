<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Badge.png" alt="Lit-GPT" width="128"/>

# ⚡ Lit-GPT

<!--
<p align="center">
  <a href="https://www.lightning.ai/">Lightning.ai</a> •
  <a href="https://lightning.ai/docs/pytorch/stable/">PyTorch Lightning</a> •
  <a href="https://lightning.ai/docs/fabric/stable/">Fabric</a>
</p>
-->

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)
![cpu-tests](https://github.com/lightning-AI/lit-stablelm/actions/workflows/cpu-tests.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM.gif" alt="Lit-GPT and pineapple pizza" width="500px"/>

</div>

&nbsp;

# ⚡ Lit-GPT

[Implementation](lit_gpt/model.py) of state-of-the-art open-source large language models released under the **Apache 2.0 license**.

The three most recently added models:

| Model                                                                                | Model size                               | Reference                                                                                                                    |
|--------------------------------------------------------------------------------------|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [Gemma](tutorials/download_gemma.md) by Google                                       | 2B, 7B                                   | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)                         |
| [Mistral and Mixtral](tutorials/download_mistral.md) by Mistral AI                   | 7B                                       | [Mistral website](https://mistral.ai/)                                                                                       |
| [Phi](tutorials/download_phi.md) by Microsoft Research                               | 1.3B, 2.7B                               | [Li et al. 2023](https://arxiv.org/abs/2309.05463)                                                                           |

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.

&nbsp;

---

**🏆 NeurIPS 2023 Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day**

The Lit-GPT repository was the official starter kit for the [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io), which is a competition focused on finetuning an existing non-instruction tuned LLM for 24 hours on a single GPU.

---

&nbsp;

## Lit-GPT design principles

This repository follows the main principle of **openness through clarity**.

**Lit-GPT** is:

- **Easy to use:** Chat, finetune, pretrain - all from the command line without writing code.
- **Correct:** Numerically equivalent to the original model.
- **Optimized:** Runs fast on consumer hardware or at scale.
- **Open-source:** No strings attached.

&nbsp;

## Installation

```bash
pip install litgpt
```

&nbsp;

## Use the model

[Choose a model](tutorials/models.md) and download its weights.

```bash
litgpt download google/gemma-2b-it
```
Chat with a model:

```bash
litgpt chat google/gemma-2b-it

>> Prompt:
```
This will run the 3B pretrained model and require ~7 GB of GPU memory using the `bfloat16` datatype.

[Full guide for generating samples from a model](tutorials/inference.md).

&nbsp;

### Run large models on smaller consumer devices

We support 4-bit quantization (as in QLoRA), (bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq) and 8-bit quantization (bnb.int8) for inference by following [this guide](tutorials/quantize.md).

&nbsp;

## Finetune the model

We provide a simple command that lets you finetune a pretrained model on common datasets such as [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) or your own data.
For example, you can either use a predefined configuration

```bash
litgpt finetune lora --config finetune/alpaca/gemma-2b.yaml
```

or choose your own parameters

```bash
litgpt finetune lora \
  --checkpoint_dir checkpoints/google/gemma-2b-it \
  --dataset Deita \
  --batch_size 32 \
  --max_epochs 2
```
The finetuning requires at least one GPU with ~12 GB memory (RTX 3060). As soon as it finishes, you can use the checkpoint:

```bash
litpgt chat out/lora/final
````

More details about each finetuning method and how you can apply it to **your own data** can be found in our technical how-to guides.

&nbsp;

### Finetuning how-to guides

These technical tutorials illustrate how to run the finetuning code.

- [Finetune with LoRA or QLoRA](tutorials/finetune_lora.md)
- [Finetune with Adapters](tutorials/finetune_adapter.md)
- [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)
- [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)

&nbsp;

## Pretraining

TODO

```bash
litgpt pretrain --config pretrain/tinyllama.yaml
```

&nbsp;


## Supported datasets

Lit-GPT includes a variety of dataset preparation scripts for finetuning and pretraining. Additional information about the datasets and dataset preparation is provided in the [Preparing Datasets](tutorials/prepare_dataset.md) tutorial.


&nbsp;

## Get involved!

[Join our Discord](https://discord.gg/VptPCZkGNa) to build high-performance, truly open-source models for the common benefit of the community.

<img align="right" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Illustration.png" alt="Lit-GPT" width="128"/>

&nbsp;

## Acknowledgements

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

&nbsp;

## Citation

If you use Lit-GPT in your research, please cite the following work:

```bibtex
@misc{lit-gpt-2023,
  author       = {Lightning AI},
  title        = {Lit-GPT},
  howpublished = {\url{https://github.com/Lightning-AI/lit-gpt}},
  year         = {2023},
}
```

&nbsp;

## License

Lit-GPT is released under the [Apache 2.0](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE) license.
