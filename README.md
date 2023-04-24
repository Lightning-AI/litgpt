<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/Lit_LLaMA_Badge3x.png" alt="Lit-LLaMA" width="128"/>

# ⚡ Lit-LLaMA ️

<!--
<p align="center">
  <a href="https://www.lightning.ai/">Lightning.ai</a> •
  <a href="https://lightning.ai/docs/pytorch/stable/">PyTorch Lightning</a> •
  <a href="https://lightning.ai/docs/fabric/stable/">Fabric</a>
</p>
-->

![cpu-tests](https://github.com/lightning-AI/lit-llama/actions/workflows/cpu-tests.yml/badge.svg) [![Build Status](https://dev.azure.com/Lightning-AI/lit%20Models/_apis/build/status%2FLightning-AI.lit-LLaMA?branchName=main)](https://dev.azure.com/Lightning-AI/lit%20Models/_build/latest?definitionId=49&branchName=main) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-llama/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/Llama_pineapple.gif" alt="Lit-LLaMA and pineapple pizza" width="500px"/>

</div>

# ⚡ Lit-LLaMA ️
Independent implementation of [LLaMA](<https://github.com/facebookresearch/llama>) that is fully open source under the **Apache 2.0 license.**

This implementation builds on [nanoGPT](<https://github.com/karpathy/nanoGPT>). Weights are distributed by Meta under a [research-only license](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md#model-details).

## Why?

We believe that AI should be fully open source and part of the collective knowledge.

The original [LLaMA code](https://github.com/facebookresearch/llama) is [GPL licensed](https://github.com/facebookresearch/llama/blob/main/LICENSE) which means any project using it must also be released under GPL.

This "taints" any other code and prevents integration with the rest of the ecosystem.

**Lit-LLaMA solves that for good.**

&nbsp;

## Design principles
**Lit-LLaMA** is:

- **Simple:** Single-file implementation without boilerplate.
- **Correct:** Numerically equivalent to the original model.
- **Optimized:** Runs on consumer hardware or at scale.
- **Open-source:** No strings attached.

## Get involved!
[Join our Discord](https://discord.gg/VptPCZkGNa) to build high-performance, truly open-source models for the common benefit of the community.

&nbsp;

## Setup

Clone the repo

```bash
git clone https://github.com/Lightning-AI/lit-llama
cd lit-llama
```

install dependencies

```bash
pip install -r requirements.txt
```

You are all set! 🎉

&nbsp;

## Use the model

To generate text predictions, you need to download the model weights. **If you don't have them, check out our [guide](howto/download_weights.md).**

Run inference:

```bash
python generate.py --prompt "Hello, my name is"
```

This will run the 7B model and require ~26 GB of GPU memory (A100 GPU).

[Full guide for generating samples from the model](howto/inference.md).

### Run Lit-LLaMA on consumer devices

On GPUs with `bfloat16` support, the `generate.py` script will automatically convert the weights and consume about ~14 GB.
For GPUs with less memory, or ones that don't support `bfloat16`, enable quantization (`--quantize llm.int8`):

```bash
python generate.py --quantize llm.int8 --prompt "Hello, my name is"
```

See `python generate.py --help` for more options.

You can also use GPTQ-style int4 quantization, but this needs conversions of the weights first:

```bash
python quantize.py --checkpoint_path lit-llama.pth --tokenizer_path tokenizer.model --output_path llama-7b-gptq.4bit.pth --dtype bfloat16  --quantize gptq.int4
```

With the generated quantized checkpoint generation works as usual with `--quantize gptq.int4`, bringing GPU usage to about ~5GB. As only the weights of the Linear layers are quantized, it is useful to use `--dtype bfloat16` even with the quantization enabled.

[Full guide for generating samples from the model](howto/inference.md).

## Finetune the model

We provide a simple training scripts in `finetune_lora.py` and `finetune_adapter.py` that instruction-tunes a pretrained model on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset using the techniques of [LoRA](https://arxiv.org/abs/2106.09685) and [Adapter](https://arxiv.org/abs/2303.16199).

1. Download the data and generate a instruction tuning dataset:

   ```bash
   python scripts/prepare_alpaca.py
   ```

2. Run the finetuning script

   ```bash
   python finetune_lora.py
   ```
   or 
   ```bash
   python finetune_adapter.py
   ```

It is expected that you have downloaded the pretrained weights as described above.
The finetuning requires at least one GPU with ~24 GB memory (GTX 3090). Follow the instructions in the script to efficiently fit your GPU memory.
Note: For some GPU models you might need to set `torch.backends.cuda.enable_flash_sdp(False)` (see comments at the top of the script).

More details about each finetuning method and how you can apply it to your own data can be found in our technical how-to guides.

### Finetuning How-To Guides

These technical tutorials illustrate how to run the finetuning code.

- [Finetune with LoRA](howto/finetune_lora.md)
- [Finetune with Adapters](howto/finetune_adapter.md)

### Understanding Finetuning -- Conceptual Tutorials

Looking for conceptual tutorials and explanations? We have some additional articles below:

- [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)

## Get involved!

We are on a quest towards fully open source AI.

<img align="right" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/Lit_LLaMA_Illustration3x.png" alt="Lit-LLaMA" width="128"/>

Join us and start contributing, especially on the following areas:

- [ ] [Pre-training](https://github.com/Lightning-AI/lit-llama/labels/pre-training)
- [ ] [Fine-tuning (full and LoRA)](https://github.com/Lightning-AI/lit-llama/labels/fine-tuning)
- [ ] [Quantization](https://github.com/Lightning-AI/lit-llama/labels/quantization)
- [ ] [Sparsification](https://github.com/Lightning-AI/lit-llama/labels/sparsification)

Look at `train.py` for a starting point towards pre-training / fine-tuning using [Lightning Fabric](https://lightning.ai/docs/fabric/stable/).

We welcome all individual contributors, regardless of their level of experience or hardware. Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment. 

Unsure about contributing? Check out our [Contributing to Lit-LLaMA: A Hitchhiker’s Guide to the Quest for Fully Open-Source AI](https://lightning.ai/pages/community/tutorial/contributing-to-lit-llama-a-hitchhikers-guide-to-the-quest-for-fully-open-source-ai/) guide.

Don't forget to [join our Discord](https://discord.gg/VptPCZkGNa)!

## Acknowledgements

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@FacebookResearch](https://github.com/facebookresearch) for the original [LLaMA implementation](https://github.com/facebookresearch/llama)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@IST-DASLab](https://github.com/IST-DASLab) for [GPTQ](https://github.com/IST-DASLab/gptq)

## License

Lit-LLaMA is released under the [Apache 2.0](https://github.com/Lightning-AI/lightning-llama/blob/main/LICENSE) license.
