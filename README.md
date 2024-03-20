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

⚡ LitGPT is a hackable [implementation](litgpt/model.py) of state-of-the-art open-source large language models released under the **Apache 2.0 license**.

&nbsp;
## LitGPT supports

✅ &nbsp;[The latest model weights](tutorials/download_model_weights.md): Gemma, Mistral, Mixtral, Phi 2, Llama 2, Falcon, CodeLlama, and [many more](tutorials/download_model_weights.md).

✅ &nbsp;Optimized and efficient code: Flash Attention v2, multi-GPU support via fully-sharded data parallelism, [optional CPU offloading](tutorials/oom.md#do-sharding-across-multiple-gpus), and [TPU and XLA support](extensions/xla).

✅ &nbsp;[Pretraining](tutorials/pretraining.md), [finetuning](tutorials/finetune.md), and [inference](tutorials/inference.md) in various precision settings: FP32, FP16, BF16, and FP16/FP32 mixed.

✅ &nbsp;[Configuration files](config_hub) for great out-of-the-box performance.

✅ &nbsp;Efficient finetuning: [LoRA](tutorials/finetune_lora.md), [QLoRA](tutorials/finetune_lora.md), [Adapter](tutorials/finetune_adapter.md), and [Adapter v2](tutorials/finetune_adapter.md).

✅ &nbsp;[Quantization](tutorials/quantize.md): 4-bit floats, 8-bit integers, and double quantization.

✅ &nbsp;[Exporting](https://github.com/Lightning-AI/litgpt/blob/wip/tutorials/convert_lit_models.md) to other popular model weight formats.

✅ &nbsp;Many popular datasets for [pretraining](tutorials/pretrain_tinyllama.md) and [finetuning](tutorials/prepare_dataset.md), and [support for custom datasets](tutorials/prepare_dataset.md#preparing-custom-datasets-for-instruction-finetuning).

✅ &nbsp;Readable and easy-to-modify code to experiment with the latest research ideas.


&nbsp;
<br>
&nbsp;

## Project templates

The following [Lightning Studio](https://lightning.ai/lightning-ai/studios) templates provide LitGPT tutorials and projects in reproducible environments with multi-GPU and multi-node support:


|                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <p align="left">[Prepare the TinyLlama 1T token dataset](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset) <br> [<img src="https://pl-public-data.s3.amazonaws.com/assets_litgpt/readme/3.webp" width="300"></p>](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset)         | [Pretrain LLMs - TinyLlama 1.1B](https://lightning.ai/lightning-ai/studios/pretrain-llms-tinyllama-1-1b) <br> <p align="left">[<img src="https://pl-public-data.s3.amazonaws.com/assets_litgpt/readme/4.webp" width="300"></p>](https://lightning.ai/lightning-ai/studios/pretrain-llms-tinyllama-1-1b)                                        |
| [Continued Pretraining with TinyLlama 1.1B](https://lightning.ai/lightning-ai/studios/continued-pretraining-with-tinyllama-1-1b) <br> <p align="left">[<img src="https://pl-public-data.s3.amazonaws.com/assets_litgpt/readme/1.webp" width="300"></p>](https://lightning.ai/lightning-ai/studios/continued-pretraining-with-tinyllama-1-1b) | [Instruction finetuning - TinyLlama 1.1B LLM](https://lightning.ai/lightning-ai/studios/instruction-finetuning-tinyllama-1-1b-llm) <br> <p align="left">[<img src="https://pl-public-data.s3.amazonaws.com/assets_litgpt/readme/2.webp" width="300"></p>](https://lightning.ai/lightning-ai/studios/instruction-finetuning-tinyllama-1-1b-llm) |
|                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                |





&nbsp;
<br>
&nbsp;



## Installing LitGPT

You can install LitGPT with all dependencies (including CLI, quantization, tokenizers for all models, etc.) using the following pip command:

```bash
pip install 'litgpt[all]'
```

Alternatively, can install litgpt from a cloned GitHub repository:

```bash
git clone https://github.com/Lightning-AI/litgpt
cd litgpt
pip install -e '.[all]'
```


&nbsp;

## Using LitGPT


Below is a minimal example to get started with the LitGPT command line interface (CLI), illustrating how to download and use a model:


```bash
# 1) Download a pretrained model
litgpt download --repo_id mistralai/Mistral-7B-Instruct-v0.2

# 2) Chat with the model
litgpt chat \
  --checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.2

>> Prompt: What do Llamas eat?
```

For more information, refer to the [download](tutorials/download_model_weights.md) and [inference](tutorials/inference.md) tutorials.


&nbsp;
## Finetuning and pretraining

LitGPT supports [pretraining](tutorials/pretrain_tinyllama.md) and [finetuning](tutorials/finetune.md) to optimize models on excisting or custom datasets. Below is an example showing how to finetune a model with LoRA:

```bash
# 1) Download a pretrained model
litgpt download --repo_id microsoft/phi-2

# 2) Finetune the model
litgpt finetune lora \
  --checkpoint_dir checkpoints/microsoft/phi-2 \
  --data Alpaca2k \
  --out_dir out/phi-2-lora

# 3) Chat with the model
litgpt chat \
  --checkpoint_dir out/phi-2-lora/final
```

&nbsp;
## Configuration files for enhanced performance

LitGPT also allows users to use configuration files in YAML format instead of specifying settings via the command line interface and comes with a set of model-specific defaults for good out-of-the-box performance:


```bash
litgpt finetune lora \
  --config https://github.com/Lightning-AI/litgpt/blob/wip/config_hub/finetune/llama-2-7b/lora.yaml
```

For added convenience, you can also manually override config file setting via the CLI:


```bash
litgpt finetune lora \
  --config https://raw.githubusercontent.com/Lightning-AI/litgpt/main/config_hub/finetune/llama-2-7b/lora.yaml \
  --lora_r 4
```

You can browse the available configuration files [here](https://github.com/Lightning-AI/litgpt/tree/main/config_hub).

&nbsp;

> [!TIP]
> **Run large models on smaller consumer devices:**
> We support 4-bit quantization (as in QLoRA), (bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq) and 8-bit quantization (bnb.int8) for inference by following [this guide](tutorials/quantize.md).


&nbsp;
<br>
&nbsp;

## Customization

LitGPT supports rich and customizable [config files](config_hub) to tailor the LLM training to your dataset and hardware needs. Shown below is a configuration file for LoRA finetuning:

```yaml
# The path to the base model's checkpoint directory to load for finetuning. (type: <class 'Path'>, default: checkpoints/stabilityai/stablelm-base-alpha-3b)
checkpoint_dir: checkpoints/meta-llama/Llama-2-7b-hf

# Directory in which to save checkpoints and logs. (type: <class 'Path'>, default: out/lora)
out_dir: out/finetune/qlora-llama2-7b

# The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true". (type: Optional[str], default: null)
precision: bf16-true

# If set, quantize the model with this algorithm. See ``tutorials/quantize.md`` for more information. (type: Optional[Literal['nf4', 'nf4-dq', 'fp4', 'fp4-dq', 'int8-training']], default: null)
quantize: bnb.nf4

# How many devices/GPUs to use. (type: Union[int, str], default: 1)
devices: 1

# The LoRA rank. (type: int, default: 8)
lora_r: 32

# The LoRA alpha. (type: int, default: 16)
lora_alpha: 16

# The LoRA dropout value. (type: float, default: 0.05)
lora_dropout: 0.05

# Whether to apply LoRA to the query weights in attention. (type: bool, default: True)
lora_query: true

# Whether to apply LoRA to the key weights in attention. (type: bool, default: False)
lora_key: false

# Whether to apply LoRA to the value weights in attention. (type: bool, default: True)
lora_value: true

# Whether to apply LoRA to the output projection in the attention block. (type: bool, default: False)
lora_projection: false

# Whether to apply LoRA to the weights of the MLP in the attention block. (type: bool, default: False)
lora_mlp: false

# Whether to apply LoRA to output head in GPT. (type: bool, default: False)
lora_head: false

# Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
data:
  class_path: litgpt.data.Alpaca2k
  init_args:
    mask_prompt: false
    val_split_fraction: 0.05
    prompt_style: alpaca
    ignore_index: -100
    seed: 42
    num_workers: 4
    download_dir: data/alpaca2k

# Training-related arguments. See ``litgpt.args.TrainArgs`` for details
train:

  # Number of optimizer steps between saving checkpoints (type: Optional[int], default: 1000)
  save_interval: 200

  # Number of iterations between logging calls (type: int, default: 1)
  log_interval: 1

  # Number of samples between optimizer steps across data-parallel ranks (type: int, default: 128)
  global_batch_size: 8

  # Number of samples per data-parallel rank (type: int, default: 4)
  micro_batch_size: 2

  # Number of iterations with learning rate warmup active (type: int, default: 100)
  lr_warmup_steps: 10

  # Number of epochs to train on (type: Optional[int], default: 5)
  epochs: 4

  # Total number of tokens to train on (type: Optional[int], default: null)
  max_tokens:

  # Limits the number of optimizer steps to run (type: Optional[int], default: null)
  max_steps:

  # Limits the length of samples (type: Optional[int], default: null)
  max_seq_length: 512

  # Whether to tie the embedding weights with the language modeling head weights (type: Optional[bool], default: null)
  tie_embeddings:

  #   (type: float, default: 0.0003)
  learning_rate: 0.0002

  #   (type: float, default: 0.02)
  weight_decay: 0.0

  #   (type: float, default: 0.9)
  beta1: 0.9

  #   (type: float, default: 0.95)
  beta2: 0.95

  #   (type: Optional[float], default: null)
  max_norm:

  #   (type: float, default: 6e-05)
  min_lr: 6.0e-05

# Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details
eval:

  # Number of optimizer steps between evaluation calls (type: int, default: 100)
  interval: 100

  # Number of tokens to generate (type: Optional[int], default: 100)
  max_new_tokens: 100

  # Number of iterations (type: int, default: 100)
  max_iters: 100

# The name of the logger to send metrics to. (type: Literal['wandb', 'tensorboard', 'csv'], default: csv)
logger_name: csv

# The random seed to use for reproducibility. (type: int, default: 1337)
seed: 1337
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

## Tutorials, how-to guides, and docs

-  Finetuning, incl. LoRA, QLoRA, and Adapters ([tutorials/finetune.md](tutorials/finetune.md))
-  Pretraining ([tutorials/pretrain_tinyllama.md](tutorials/pretrain_tinyllama.md))
-  Model evaluation ([tutorials/evaluation.md](tutorials/evaluation.md))
-  Supported and custom datasets ([tutorials/prepare_dataset.md](tutorials/prepare_dataset.md))
-  Quantization ([tutorials/quantize.md](tutorials/quantize.md))
-  Tips for dealing with out-of-memory (OOM) errors ([tutorials/oom.md](tutorials/oom.md))





&nbsp;

## XLA

Lightning AI has partnered with Google to add first-class support for [Cloud TPUs](https://cloud.google.com/tpu) in [Lightning’s frameworks](https://github.com/Lightning-AI/lightning) and LitGPT,
helping democratize AI for millions of developers and researchers worldwide.

Using TPUs with Lightning is as straightforward as changing one line of code.

We provide scripts fully optimized for TPUs in the [XLA directory](extensions/xla).



&nbsp;

## Acknowledgements

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and the [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)


&nbsp;


## Community showcase

Check out the projects below using and building on LitGPT. If you have a project you'd like to add to this section, please don't hestiate to open a pull request.

&nbsp;

**🏆 NeurIPS 2023 Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day**

The LitGPT repository was the official starter kit for the [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io), which is a competition focused on finetuning an existing non-instruction tuned LLM for 24 hours on a single GPU.

&nbsp;

**🦙 TinyLlama: An Open-Source Small Language Model**

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

