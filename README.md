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

<<<<<<< Updated upstream
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM.gif" alt="Lit-GPT and pineapple pizza" width="500px"/>

=======
>>>>>>> Stashed changes
</div>

&nbsp;

<<<<<<< Updated upstream
# ⚡ Lit-GPT

[Implementation](lit_gpt/model.py) of state-of-the-art open-source large language models released under the **Apache 2.0 license**.

The three most recently added models:

| Model                                                                                | Model size                               | Reference                                                                                                                    |
|--------------------------------------------------------------------------------------|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [Gemma](tutorials/download_gemma.md) by Google                                       | 2B, 7B                                   | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)                         |
| [Mistral and Mixtral](tutorials/download_mistral.md) by Mistral AI                   | 7B                                       | [Mistral website](https://mistral.ai/)                                                                                       |
| [Phi](tutorials/download_phi.md) by Microsoft Research                               | 1.3B, 2.7B                               | [Li et al. 2023](https://arxiv.org/abs/2309.05463)                                                                           |

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.
=======
⚡ LitGPT is a hackable [implementation](litgpt/model.py) of state-of-the-art open-source large language models released under the **Apache 2.0 license**. It supports a large number of pretrained models. The three most recently added models are shown below:
>>>>>>> Stashed changes

&nbsp;

| Model                                                                                | Model size                               | Reference                                                                                                                    |
|--------------------------------------------------------------------------------------|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [Gemma](tutorials/download_gemma.md) by Google                                       | 2B, 7B                                   | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)                         |
| [Mistral and Mixtral](tutorials/download_mistral.md) by Mistral AI                   | 7B                                       | [Mistral website](https://mistral.ai/)                                                                                       |
| [Phi](tutorials/download_phi.md) by Microsoft Research                               | 1.3B, 2.7B                               | [Li et al. 2023](https://arxiv.org/abs/2309.05463)                                                                           |

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.

<<<<<<< Updated upstream
The Lit-GPT repository was the official starter kit for the [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io), which is a competition focused on finetuning an existing non-instruction tuned LLM for 24 hours on a single GPU.
=======
> [!NOTE] 
> For a complete list of all supported models, please refer to the [Download Model Weights with LitGPT](tutorials/download_model_weights.md) tutorial.
>>>>>>> Stashed changes

&nbsp;

&nbsp;

## Getting Started in 3 Steps

Below is a minimal example to get started with the LitGPT command line interface (CLI), illustrating how to download a model, optionally finetune it using low-rank adaptation (LoRA), and start chatting with it:


```bash
# 1) Download a pretrained model
python litgpt download --repo_id mistralai/Mistral-7B-v0.1

# 2) Optionally finetune the model
python litgpt finetune lora \
  --checkpoint_dir checkpoints/mistralai/Mistral-7B-v0.1 \
  --train.micro_batch_size 2 \
  --lora_r 4 \
  --precision bf16-true \
  --data Alpaca2k \
  --out_dir out/my-finetuned-model

# 3) Chat with the model
python finetune chat \
  --checkpoint_dir checkpoints/mistralai/Mistral-7B-v0.1 \
  --data Alpaca2k

>> Prompt:
```

For more information, refer to the [download](tutorials/download_model_weights.md), [pretraining](tutorials/pretrain_tinyllama.md), [finetuning](tutorials/finetune_lora.md), and [inference](tutorials/inference.md) tutorials.


&nbsp;

## Configuration Files for Enhanced Performance

LitGPT also allows users to use configuration files in YAML format instead of specifying settings via the command line interface and comes with a set of model-specific defaults for good out-of-the-box performance:


```bash
python litgpt finetune lora \
  --config https://github.com/Lightning-AI/litgpt/blob/wip/config_hub/finetune/llama-2-7b/lora.yaml
```

For added convenience, you can also manually override config file setting via the CLI:


```bash
python litgpt finetune lora 
  --config https://github.com/Lightning-AI/litgpt/blob/wip/config_hub/finetune/llama-2-7b/lora.yaml \
  --lora_r 4
```

You can browse the available configuration files [here](https://github.com/Lightning-AI/litgpt/tree/main/config_hub).

&nbsp;

> [!TIP] 
> **Run large models on smaller consumer devices**
> We support 4-bit quantization (as in QLoRA), (bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq) and 8-bit quantization (bnb.int8) for inference by following [this guide](tutorials/quantize.md).


&nbsp;
<br>
&nbsp;

## Installing LitGPT

You can install LitGPT with all dependencies (including CLI, quantization, tokenizers for all models, etc.) using the following pip command:

```bash
pip install "litgpt[all]"
```

Alternatively, can install litgpt from a cloned GitHub repository:

```bash
git clone https://github.com/Lightning-AI/litgpt/blob/wip/config_hub/finetune/llama-2-7b/full.yaml
cd litgpt
pip install ".[all]"
```

&nbsp;

## Lit-GPT design principles

This repository follows the main principle of **openness through clarity**.

**Lit-GPT** is:

- **Easy to use:** Chat, finetune, pretrain - all from the command line without writing code.
- **Correct:** Numerically equivalent to the original model.
- **Optimized:** Runs fast on consumer hardware or at scale.
- **Open-source:** No strings attached.

&nbsp;

<<<<<<< Updated upstream
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
=======
## Get involved!

We appreciate your feedback and contributions. If you have feature requests, questions, or want to contribute code or config files, please don't hesitate to use the [GitHub Issue](https://github.com/Lightning-AI/litgpt/issues) tracker.

We welcome all individual contributors, regardless of their level of experience or hardware. Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment.

&nbsp;

> [!TIP] 
> Unsure about contributing? Check out our [How to Contribute to LitGPT](https://lightning.ai/pages/community/tutorial/how-to-contribute-to-litgpt/) guide.

If you have general questions about building with LitGPT, please [join our Discord](https://discord.gg/VptPCZkGNa).
>>>>>>> Stashed changes

```bash
litgpt pretrain --config pretrain/tinyllama.yaml
```

&nbsp;

## Tutorials and how-to guides

- [Finetuning (incl. LoRA, QLoRA, and Adapters)](tutorials/finetune.md)
- [Pretraining](tutorials/pretrain_tinyllama.md)
- [Model evaluation](tutorials/evaluation.md)
- [Supported and custom datasets](tutorials/prepare_dataset.md)
- [Quantization](tutorials/quantize.md)
- [Tips for dealing with out-of-memory (OOM) errors](tutorials/oom.md)



<<<<<<< Updated upstream
Lit-GPT includes a variety of dataset preparation scripts for finetuning and pretraining. Additional information about the datasets and dataset preparation is provided in the [Preparing Datasets](tutorials/prepare_dataset.md) tutorial.
=======
>>>>>>> Stashed changes


&nbsp;

<<<<<<< Updated upstream
## Get involved!

[Join our Discord](https://discord.gg/VptPCZkGNa) to build high-performance, truly open-source models for the common benefit of the community.

<img align="right" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Illustration.png" alt="Lit-GPT" width="128"/>
=======
## XLA

Lightning AI has partnered with Google to add first-class support for [Cloud TPUs](https://cloud.google.com/tpu) in [Lightning’s frameworks](https://github.com/Lightning-AI/lightning) and LitGPT,
helping democratize AI for millions of developers and researchers worldwide.

Using TPUs with Lightning is as straightforward as changing one line of code.

We provide scripts fully optimized for TPUs in the [XLA directory](xla)


>>>>>>> Stashed changes

&nbsp;

## Acknowledgements

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)


&nbsp;

## Projects using LitGPT


**🏆 NeurIPS 2023 Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day**

The LitGPT repository was the official starter kit for the [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io), which is a competition focused on finetuning an existing non-instruction tuned LLM for 24 hours on a single GPU.

&nbsp;

**TinyLlama: An Open-Source Small Language Model**

LitGPT powered the [TinyLlama project](https://github.com/jzhang38/TinyLlama) and [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385) research paper.


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

<<<<<<< Updated upstream
Lit-GPT is released under the [Apache 2.0](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE) license.
=======
LitGPT is released under the [Apache 2.0](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE) license.

>>>>>>> Stashed changes
