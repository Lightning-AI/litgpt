# Pretrain LLMs with LitGPT


The simplest way to get started with pretraining LLMs in LitGPT ...


&nbsp;
## The Pretraining API

You can pretrain models in LitGPT using the `litgpt pretrain` API starting with any of the available architectures listed by calling `litgpt pretrain` without any additional arguments:

```bash
litgpt pretrain
```

Shown below is an abbreviated list

```
ValueError: Please specify --model_name <model_name>. Available values:
Camel-Platypus2-13B
...
Gemma-2b
...
Llama-2-7b-hf
...
Mixtral-8x7B-v0.1
...
pythia-14m
```

For demonstration purposes, we can pretrain a small 14 million-parameter Pythia model on the small TinyStories dataset using the [debug.yaml config file](https://github.com/Lightning-AI/litgpt/blob/main/config_hub/pretrain/debug.yaml) as follows:

```bash
litgpt pretrain \
   --model_name pythia-14m \
   --config https://raw.githubusercontent.com/Lightning-AI/litgpt/main/config_hub/pretrain/debug.yaml
```




&nbsp;
## Pretrain a 1.1B TinyLlama model

You can find an end-to-end LitGPT tutorial for pretraining a TinyLlama model using LitGPT [here](pretrain_tinyllama.md).


&nbsp;
## Optimize LitGPT pretraining with Lightning Thunder

[Lightning Thunder](https://github.com/Lightning-AI/lightning-thunder) is a source-to-source compiler for PyTorch, which is fully compatible with LitGPT. In experiments, Thunder resulted in a 40% speed-up compared to using regular PyTorch when finetuning a 7B Llama 2 model.

For more information, see the [Lightning Thunder extension README](https://github.com/Lightning-AI/lightning-thunder).


&nbsp;
## Project templates

The following [Lightning Studio](https://lightning.ai/lightning-ai/studios) templates provide LitGPT pretraining projects in reproducible environments with multi-GPU and multi-node support:
&nbsp;

|                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <p align="left">[Prepare the TinyLlama 1T token dataset](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset) <br> [<img src="https://pl-public-data.s3.amazonaws.com/assets_litgpt/readme/3.webp" width="300"></p>](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset)         | [Pretrain LLMs - TinyLlama 1.1B](https://lightning.ai/lightning-ai/studios/pretrain-llms-tinyllama-1-1b) <br> <p align="left">[<img src="https://pl-public-data.s3.amazonaws.com/assets_litgpt/readme/4.webp" width="300"></p>](https://lightning.ai/lightning-ai/studios/pretrain-llms-tinyllama-1-1b)                                        |
| [Continued Pretraining with TinyLlama 1.1B](https://lightning.ai/lightning-ai/studios/continued-pretraining-with-tinyllama-1-1b) <br> <p align="left">[<img src="https://pl-public-data.s3.amazonaws.com/assets_litgpt/readme/1.webp" width="300"></p>](https://lightning.ai/lightning-ai/studios/continued-pretraining-with-tinyllama-1-1b) | |
|                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                     