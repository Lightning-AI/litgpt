<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Badge.png" alt="LitGPT" width="128"/>

# ⚡ LitGPT

<!--
<p align="center">
  <a href="https://www.lightning.ai/">Lightning.ai</a> •
  <a href="https://lightning.ai/docs/pytorch/stable/">PyTorch Lightning</a> •
  <a href="https://lightning.ai/docs/fabric/stable/">Fabric</a>
</p>
-->

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)
![cpu-tests](https://github.com/lightning-AI/lit-stablelm/actions/workflows/cpu-tests.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)

</div>

&nbsp;

⚡ LitGPT is a hackable [implementation](litgpt/model.py) of state-of-the-art open-source large language models released under the **Apache 2.0 license**. It supports a large number of pretrained models. The three most recently added models are shown below:

&nbsp;

| Model                                                                                | Model size                               | Reference                                                                                                                    |
|--------------------------------------------------------------------------------------|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [Gemma](tutorials/download_gemma.md) by Google                                       | 2B, 7B                                   | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)                         |
| [Mistral and Mixtral](tutorials/download_mistral.md) by Mistral AI                   | 7B                                       | [Mistral website](https://mistral.ai/)                                                                                       |
| [Phi](tutorials/download_phi.md) by Microsoft Research                               | 1.3B, 2.7B                               | [Li et al. 2023](https://arxiv.org/abs/2309.05463)                                                                           |



> [!NOTE] 
> For a complete list of all supported models, please refer to the [Download Model Weights with LitGPT](tutorials/download_model_weights.md) tutorial.

&nbsp;

&nbsp;

## Getting started in 3 steps

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

## Configuration files for enhanced performance

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

## LitGPT design principles

This repository follows the main principle of **openness through clarity**.

**LitGPT** is:

- **Simple:** Single-file implementation without boilerplate.
- **Correct:** Numerically equivalent to the original model.
- **Optimized:** Runs fast on consumer hardware or at scale.
- **Open-source:** No strings attached.

Avoiding code duplication is **not** a goal. **Readability** and **hackability** are.

&nbsp;

## Get involved!

We appreciate your feedback and contributions. If you have feature requests, questions, or want to contribute code or config files, please don't hesitate to use the [GitHub Issue](https://github.com/Lightning-AI/litgpt/issues) tracker.

We welcome all individual contributors, regardless of their level of experience or hardware. Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment.

&nbsp;

> [!TIP] 
> Unsure about contributing? Check out our [How to Contribute to LitGPT](https://lightning.ai/pages/community/tutorial/how-to-contribute-to-litgpt/) guide.

If you have general questions about building with LitGPT, please [join our Discord](https://discord.gg/VptPCZkGNa).


&nbsp;

## Tutorials and how-to guides

- [Finetuning (incl. LoRA, QLoRA, and Adapters)](tutorials/finetune.md)
- [Pretraining](tutorials/pretrain_tinyllama.md)
- [Model evaluation](tutorials/evaluation.md)
- [Supported and custom datasets](tutorials/prepare_dataset.md)
- [Quantization](tutorials/quantize.md)
- [Tips for dealing with out-of-memory (OOM) errors](tutorials/oom.md)





&nbsp;

## XLA

Lightning AI has partnered with Google to add first-class support for [Cloud TPUs](https://cloud.google.com/tpu) in [Lightning’s frameworks](https://github.com/Lightning-AI/lightning) and LitGPT,
helping democratize AI for millions of developers and researchers worldwide.

Using TPUs with Lightning is as straightforward as changing one line of code.

We provide scripts fully optimized for TPUs in the [XLA directory](xla)



&nbsp;

## Acknowledgements

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)


&nbsp;

## LitGPT Studios

The following [Lightning Studio](https://lightning.ai/lightning-ai/studios) templates provide LitGPT tutorials and projects in reproducible environments with multi-GPU and multi-node support:

| [Continued Pretraining with TinyLlama 1.1B](https://lightning.ai/lightning-ai/studios/continued-pretraining-with-tinyllama-1-1b) <br> [<img src="https://grid-cloud-litng-ai-03.s3.amazonaws.com/projects/01h5rahq6gbhw5m4bzyws0at5h/cloudspaces/01hq0v57k4zeswpcx1nwaj9ck9/thumbnails/image.jpeg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24LZGPJHU3I6BDPF%2F20240313%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240313T141203Z&X-Amz-Expires=3600&X-Amz-Security-Token=FwoGZXIvYXdzEGgaDEkXJ7sroCiJUMpjHCLDATYRVfka8dfeXYhLHqWcO89RaRT4l1oAXcG2QOOhFQepPb80CWILvo2L52hgr1rnG0gwWb9PB9aPcUHZVLxwwoybMTo5dN0qv7ukYpViYqKLOuthMf8KAE1xDnyHgCFprcLwGHexExH3zA1YPi8zzTonB7xkYf0MUqNqi44C9xAmRLGdofZFkR0Rg5l40lvIgU6fHCt7Gg8xWEQEQqnSoLjsDe5GfjpVatNdHFN2aNXWDO5Vcl7LF7FjSokaRI4hJXOwgCiz6MavBjIthd0ZFxiDIGHrwRN6KldDSlW5jPjdZ8QNAXrnfBxfH7iinDDP6RKhfUcu38uc&X-Amz-SignedHeaders=host&X-Amz-Signature=12a28a4a2c58de74682e91b41a5c76cb6760a6a67917f6133848948ab58d0784" width="300">](https://lightning.ai/lightning-ai/studios/continued-pretraining-with-tinyllama-1-1b) | [Instruction finetuning - TinyLlama 1.1B LLM](https://lightning.ai/lightning-ai/studios/instruction-finetuning-tinyllama-1-1b-llm) <br> [<img src="https://grid-cloud-litng-ai-03.s3.amazonaws.com/projects/01h5rahq6gbhw5m4bzyws0at5h/cloudspaces/01hn2ym8r666m28me70cwmfkg1/thumbnails/image.jpeg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24LZGPJHU3I6BDPF%2F20240313%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240313T141221Z&X-Amz-Expires=3600&X-Amz-Security-Token=FwoGZXIvYXdzEGgaDEkXJ7sroCiJUMpjHCLDATYRVfka8dfeXYhLHqWcO89RaRT4l1oAXcG2QOOhFQepPb80CWILvo2L52hgr1rnG0gwWb9PB9aPcUHZVLxwwoybMTo5dN0qv7ukYpViYqKLOuthMf8KAE1xDnyHgCFprcLwGHexExH3zA1YPi8zzTonB7xkYf0MUqNqi44C9xAmRLGdofZFkR0Rg5l40lvIgU6fHCt7Gg8xWEQEQqnSoLjsDe5GfjpVatNdHFN2aNXWDO5Vcl7LF7FjSokaRI4hJXOwgCiz6MavBjIthd0ZFxiDIGHrwRN6KldDSlW5jPjdZ8QNAXrnfBxfH7iinDDP6RKhfUcu38uc&X-Amz-SignedHeaders=host&X-Amz-Signature=7eec37bb522a47a1f55e8e99cbd05087ba351fc2b6fd8c014b049ef7c5d9b9dd" width="300">](https://lightning.ai/lightning-ai/studios/instruction-finetuning-tinyllama-1-1b-llm) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Prepare the TinyLlama 1T token dataset](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset) <br> [<img src="https://grid-cloud-litng-ai-03.s3.amazonaws.com/projects/01hg0zg8fyybp7p1sma6g9dkzm/cloudspaces/01hme55ecwaa960f1taz10swyf/thumbnails/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24LZGPJH3IOB3MP2%2F20240313%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240313T141228Z&X-Amz-Expires=3600&X-Amz-Security-Token=FwoGZXIvYXdzEGgaDKbEYqVo4gIjFcagHCLDAcW0y5rBD5EWYZ0Agb9R%2B04LO4aJEphNPCvEnwDcudEA3H%2FhKY47s1tkq0M1JeZEv6vBu2PrzHRt8fyZMcGOcP0HKfekjlI90HmJXwbNryILmA%2BrXIw0LC%2FJ6iE4Xa1um1Lm1ULxiS0MZnIyW8%2FN4mhRqXEBaq%2BjJFdE%2BV1v83XhztVr%2BGsZsjvjlAmid39KOuzZdEFbDuoi0cg5MK4NsBaA3wWJnsoZpFZc3EJHuPDowLGX1rXOj01Astu5P7vII41Hpii66MavBjItxSxO8rTjiPjg4%2FM4Q3mUZarMZZA8PMzIPX3xEO2qDu3GYabFrdG3loEeWwqT&X-Amz-SignedHeaders=host&X-Amz-Signature=90102f3cc85158c72e90092430822dbd0c73496b849dabf7a18206ca49edb7a1" width="300">](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset) | [Pretrain LLMs - TinyLlama 1.1B](https://lightning.ai/lightning-ai/studios/pretrain-llms-tinyllama-1-1b) <br> [<img src="https://grid-cloud-litng-ai-03.s3.amazonaws.com/projects/01h6ra4cyxjj75f2g2wwkx19gv/cloudspaces/01hkmc5x1zy6pdn1jde5vc4g1c/thumbnails/image.jpeg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24LZGPJH3X7JZNH7%2F20240313%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240313T141231Z&X-Amz-Expires=3600&X-Amz-Security-Token=FwoGZXIvYXdzEGgaDOO7wp1vKOj5j8WlQiLDAasfBtpnkKf1ewczzKbHlFBOUuL9EdpRK35HHAGXHxJr49Dn%2FQSFamQSFe8TfFf1qI66vv6gzAYz2zQyp%2F7JtXsnVhaA0I38RTv7RbOnlEd84EYZLgLjhhLT1QAgtnlj5GyK4MJpRbI2po%2F4ikPcBLOK3Fo7%2Bp2j7sShg%2FPW%2FzwW0guLcSULwZX2DqcED0nEEAbglXNCFzyJflDO%2FOJ6MYxotKP4UjYVjPuoXcY33%2FD%2BFTE7ZdIB%2FShgFqbHN8MHXCuQIiim6MavBjItouSoiQXrr%2FK%2BvP19AtWpahjmYc872rikZULCPSieTgPRtzwmf1DmQ0bFdKh1&X-Amz-SignedHeaders=host&X-Amz-Signature=ce69f7c68d41cd28bb270a7d06d7e3581d08da5e3e6228f439a45eb61e66d35b" width="300">](https://lightning.ai/lightning-ai/studios/pretrain-llms-tinyllama-1-1b) |


&nbsp;
<br>
&nbsp;

## Other Projects using LitGPT


**🏆 NeurIPS 2023 Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day**

The LitGPT repository was the official starter kit for the [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io), which is a competition focused on finetuning an existing non-instruction tuned LLM for 24 hours on a single GPU.

&nbsp;

**TinyLlama: An Open-Source Small Language Model**

LitGPT powered the [TinyLlama project](https://github.com/jzhang38/TinyLlama) and [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385) research paper.


&nbsp;

## Citation

If you use LitGPT in your research, please cite the following work:

```bibtex
@misc{litgpt-2023,
  author       = {Lightning AI},
  title        = {LitGPT},
  howpublished = {\url{https://github.com/Lightning-AI/litgpt}},
  year         = {2023},
}
```

&nbsp;

## License

LitGPT is released under the [Apache 2.0](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE) license.

