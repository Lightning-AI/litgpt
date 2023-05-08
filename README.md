<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Badge.png" alt="Lit-StableLM" width="128"/>

# ⚡ Lit-StableLM

<!--
<p align="center">
  <a href="https://www.lightning.ai/">Lightning.ai</a> •
  <a href="https://lightning.ai/docs/pytorch/stable/">PyTorch Lightning</a> •
  <a href="https://lightning.ai/docs/fabric/stable/">Fabric</a>
</p>
-->

![cpu-tests](https://github.com/lightning-AI/lit-stablelm/actions/workflows/cpu-tests.yml/badge.svg) <!-- [![Build Status](https://dev.azure.com/Lightning-AI/lit%20Models/_apis/build/status%2FLightning-AI.lit-StableLM?branchName=main)](https://dev.azure.com/Lightning-AI/lit%20Models/_build/latest?definitionId=49&branchName=main) --> [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM.gif" alt="Lit-StableLM and pineapple pizza" width="500px"/>

</div>

# ⚡ Lit-StableLM

Hackable implementation of state-of-the-art open-source large language models:

- StabilityAI [StableLM](https://github.com/Stability-AI/StableLM)
- EleutherAI [Pythia](https://github.com/EleutherAI/pythia)
- Together [RedPajama-INCITE](https://www.together.xyz/blog/redpajama-models-v1)

released under the **Apache 2.0 license**.

This implementation builds on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡.

Weights can be downloaded following these instructions:

- [StableLM](howto/download_stablelm.md)
- [Pythia](howto/download_pythia.md)
- [Redpajama-INCITE](howto/download_redpajama_incite.md)

## Design principles

This repository follows the main principle of **openness through clarity**.

**Lit-StableLM** is:

- **Simple:** Single-file implementation without boilerplate.
- **Correct:** Numerically equivalent to the original model.
- **Optimized:** Runs on consumer hardware or at scale.
- **Open-source:** No strings attached.

Avoiding code duplication is **not** a goal. **Readability** and **hackability** are.

## Get involved!

[Join our Discord](https://discord.gg/VptPCZkGNa) to build high-performance, truly open-source models for the common benefit of the community.

&nbsp;

## Setup

Clone the repo

```bash
git clone https://github.com/Lightning-AI/lit-stablelm
cd lit-stablelm
```

install dependencies

```bash
pip install -r requirements.txt
```

You are all set! 🎉

&nbsp;

## Use the model

To generate text predictions, you need to download the model weights. **If you don't have them, check out our [guide](howto/download_stablelm.md).**

Run inference:

```bash
python generate.py --prompt "Hello, my name is"
```

This will run the 3B pre-trained model and require ~7 GB of GPU memory using the `bfloat16` datatype.

[Full guide for generating samples from the model](howto/inference.md).

You can also chat with the model interactively:

```bash
python chat.py
```

### Run Lit-StableLM on smaller consumer devices

Porting from Lit-LLaMA in progress 👷

## Finetune the model

We provide a simple training script `finetune_adapter.py` that instruction-tunes a pretrained model on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.

1. Download the data and generate an instruction tuning dataset:

```bash
python scripts/prepare_alpaca.py
```

2. Run the finetuning script

[Adapter](https://arxiv.org/abs/2303.16199):

```bash
python finetune_adapter.py
```

The finetuning requires at least one GPU with ~12 GB memory (GTX 3060).
It is expected that you have downloaded the pretrained weights as described above.
More details about each finetuning method and how you can apply it to your own data can be found in our technical how-to guides.

### Finetuning How-To Guides

These technical tutorials illustrate how to run the finetuning code.

- [Finetune with Adapters](howto/finetune_adapter.md)

### Understanding Finetuning -- Conceptual Tutorials

Looking for conceptual tutorials and explanations? We have some additional articles below:

- [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)

## Pre-training

Porting from Lit-LLaMA in progress 👷

## Get involved!

We are on a quest towards fully open source AI.

<img align="right" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Illustration.png" alt="Lit-StableLM" width="128"/>

Join us and start contributing, especially on the following areas:

- [ ] [Pre-training](https://github.com/Lightning-AI/lit-stablelm/labels/pre-training)
- [ ] [Fine-tuning (full and LoRA)](https://github.com/Lightning-AI/lit-stablelm/labels/fine-tuning)
- [ ] [Quantization](https://github.com/Lightning-AI/lit-stablelm/labels/quantization)
- [ ] [Sparsification](https://github.com/Lightning-AI/lit-stablelm/labels/sparsification)

We welcome all individual contributors, regardless of their level of experience or hardware. Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment. 

Unsure about contributing? Check out our [Contributing to Lit-LLaMA: A Hitchhiker’s Guide to the Quest for Fully Open-Source AI](https://lightning.ai/pages/community/tutorial/contributing-to-lit-llama-a-hitchhikers-guide-to-the-quest-for-fully-open-source-ai/) guide. The same guidelines apply to Lit-StableLM.

Don't forget to [join our Discord](https://discord.gg/VptPCZkGNa)!

## Acknowledgements

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/karpathy) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@IST-DASLab](https://github.com/IST-DASLab) for [GPTQ](https://github.com/IST-DASLab/gptq)

## License

Lit-StableLM is released under the [Apache 2.0](https://github.com/Lightning-AI/lit-stablelm/blob/main/LICENSE) license.
