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

![cpu-tests](https://github.com/lightning-AI/lit-stablelm/actions/workflows/cpu-tests.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM.gif" alt="Lit-GPT and pineapple pizza" width="500px"/>

</div>

&nbsp;

# ⚡ Lit-GPT

Hackable [implementation](lit_gpt/model.py) of state-of-the-art open-source large language models released under the **Apache 2.0 license**.

Supports the following popular model checkpoints:

| Model and usage                                                     | Reference                                                                                        |
|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Meta AI [Llama 2](tutorials/download_llama_2.md)                    | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288)                                          |
| Stability AI [FreeWilly2](tutorials/download_freewilly_2.md)        | [Stability AI 2023](https://stability.ai/blog/stable-beluga-large-instruction-fine-tuned-models) |
| TII UAE [Falcon](tutorials/download_falcon.md)                      | [TII 2023](https://falconllm.tii.ae)                                                             |
| OpenLM Research [OpenLLaMA](tutorials/download_openllama.md)        | [Geng & Liu 2023](https://github.com/openlm-research/open_llama)                                 |
| LMSYS [Vicuna](tutorials/download_vicuna.md)                        | [Li et al. 2023](https://lmsys.org/blog/2023-06-29-longchat)                                     |
| Together [RedPajama-INCITE](tutorials/download_redpajama_incite.md) | [Together 2023](https://together.ai/blog/redpajama-models-v1)                                    |
| EleutherAI [Pythia](tutorials/download_pythia.md)                   | [Biderman et al. 2023](https://arxiv.org/abs/2304.01373)                                         |
| StabilityAI [StableLM](tutorials/download_stablelm.md)              | [Stability AI 2023](https://github.com/Stability-AI/StableLM)                                    |

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.

&nbsp;

---

**🏆 NeurIPS 2023 Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day**

The Lit-GPT repository is the official starter kit for the [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io), which is a competition focused on finetuning an existing non-instruction tuned LLM for 24 hours on a single GPU. The competition has two tracks, one for the A100 and another for the 4090 GPUs.

If you are interested in participating, you can learn more about the NeurIPS LLM Efficiency Challenge on the official website [here](https://llm-efficiency-challenge.github.io). Also see the [Lit-GPT NeurIPS Challenge Quickstart Guide](tutorials/neurips_challenge_quickstart.md) for helpful tips.

**The submission deadline is Oct 15th, 2023.**

---


&nbsp;


## Lit-GPT design principles

This repository follows the main principle of **openness through clarity**.

**Lit-GPT** is:

- **Simple:** Single-file implementation without boilerplate.
- **Correct:** Numerically equivalent to the original model.
- **Optimized:** Runs fast on consumer hardware or at scale.
- **Open-source:** No strings attached.

Avoiding code duplication is **not** a goal. **Readability** and **hackability** are.

&nbsp;

## Get involved!

[Join our Discord](https://discord.gg/VptPCZkGNa) to build high-performance, truly open-source models for the common benefit of the community.

&nbsp;

## Setup

Clone the repo

```bash
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
```

Lit-GPT currently relies on flash attention from PyTorch nightly. Until PyTorch 2.1 is released you'll need to install nightly manually.
Luckily this is straightforward, as shown below.

&nbsp;

**On CUDA**

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
```

**On CPU (incl Macs)**

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cpu --pre 'torch>=2.1.0dev'
```

**(Optional) install Flash Attention 2**

```bash
MAX_JOBS=4 pip install 'flash-attn>=2.0.0.post1' --no-build-isolation
```

All good, now install the dependencies plus some optional ones:

```bash
pip install -r requirements.txt tokenizers sentencepiece
```

You are all set! 🎉

&nbsp;

## Use the model

To generate text predictions, you need to download the model weights. **If you don't have them, check out our [guide](tutorials/download_stablelm.md).**

Run inference:

```bash
python generate/base.py --prompt "Hello, my name is"
```

This will run the 3B pre-trained model and require ~7 GB of GPU memory using the `bfloat16` datatype.

[Full guide for generating samples from the model](tutorials/inference.md).

You can also chat with the model interactively:

```bash
python chat/base.py
```

&nbsp;

### Run large models on smaller consumer devices

We support 4-bit quantization (as in QLoRA), LLM.int8, and GPTQ.int4 inference by following [this guide](tutorials/quantize.md).

&nbsp;

## Finetune the model

We provide a simple training scripts (`finetune/adapter.py`, `finetune/adapter_v2.py`, and `finetune/lora.py`) that instruction-tunes a pretrained model on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.

1. Download the data and generate an instruction tuning dataset:

```bash
python scripts/prepare_alpaca.py
```

2. Run the finetuning script

For example, you can either use

Adapter ([Zhang et al. 2023](https://arxiv.org/abs/2303.16199)):

```bash
python finetune/adapter.py
```

or Adapter v2 ([Gao et al. 2023](https://arxiv.org/abs/2304.15010)):

```bash
python finetune/adapter_v2.py
```

or LoRA ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)):

```bash
python finetune/lora.py
```

(Please see the [tutorials/finetune_adapter](tutorials/finetune_adapter.md) for details on the differences between the two adapter methods.)

The finetuning requires at least one GPU with ~12 GB memory (RTX 3060).

It is expected that you have downloaded the pretrained weights as described above.
More details about each finetuning method and how you can apply it to your own data can be found in our technical how-to guides.

&nbsp;

### Finetuning How-To Guides

These technical tutorials illustrate how to run the finetuning code.

- [Finetune with Adapters](tutorials/finetune_adapter.md)
- [Finetune with LoRA](tutorials/finetune_lora.md)

&nbsp;

### Understanding Finetuning -- Conceptual Tutorials

Looking for conceptual tutorials and explanations? We have some additional articles below:

- [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)

- [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)

&nbsp;

## Pre-training

Porting from Lit-LLaMA in progress 👷

&nbsp;

## Get involved!

We are on a quest towards fully open source AI.

<img align="right" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Illustration.png" alt="Lit-GPT" width="128"/>

Join us and start contributing, especially on the following areas:

- [ ] [Pre-training](https://github.com/Lightning-AI/lit-gpt/labels/pre-training)
- [ ] [Fine-tuning](https://github.com/Lightning-AI/lit-gpt/labels/fine-tuning)
- [ ] [Quantization](https://github.com/Lightning-AI/lit-gpt/labels/quantization)
- [ ] [Sparsification](https://github.com/Lightning-AI/lit-gpt/labels/sparsification)

We welcome all individual contributors, regardless of their level of experience or hardware. Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment.

Unsure about contributing? Check out our [How to Contribute to Lit-GPT and Lit-LLaMA
](https://lightning.ai/pages/community/tutorial/how-to-contribute-to-litgpt/) guide.

Don't forget to [join our Discord](https://discord.gg/VptPCZkGNa)!

&nbsp;

## Acknowledgements

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@IST-DASLab](https://github.com/IST-DASLab) for [GPTQ](https://github.com/IST-DASLab/gptq)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

&nbsp;

## License

Lit-GPT is released under the [Apache 2.0](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE) license.
