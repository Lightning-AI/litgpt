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

![cpu-tests](https://github.com/lightning-AI/lit-llama/actions/workflows/cpu-tests.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-llama/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)

</div>

# ⚡ Lit-LLaMA ️
Independent implementation of [LLaMA](<https://github.com/facebookresearch/llama>) that is fully open source under the **Apache 2.0 license.**

This implementation builds on [nanoGPT](<https://github.com/karpathy/nanoGPT>).

## Why?
The original [LLaMA code](https://github.com/facebookresearch/llama) is [GPL licensed](https://github.com/facebookresearch/llama/blob/main/LICENSE) which means any project using it must also be released under GPL.

This "taints" any other code and prevents meaningful academic and commercial use.

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

## Use the model

To generate text predictions, download the model weights following the instructions on the official [LLaMA repository](https://github.com/facebookresearch/llama). Now you should have a folder like this:

```text
checkpoints/llama
├── 7B
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   └── params.json
├── 13B
│   ...
├── tokenizer_checklist.chk
└── tokenizer.model
```

Convert the weights to the Lit-LLaMA format:

```bash
python scripts/convert_checkpoint.py \
    --output_dir checkpoints/lit-llama \
    --ckpt_dir checkpoints/llama \
    --tokenizer_path checkpoints/llama/tokenizer.model \
    --model_size 7B
```

Run inference:

```bash
python generate.py --prompt "Hello, my name is"
```

This will run the 7B model and require ~26 GB of GPU memory (A100 GPU).

### Run Lit-LLaMA on consumer devices

For GPUs with less memory, enable quantization (`--quantize true`). This will take longer to load but require ~8GB of memory.
This can run on any consumer GPU.

```bash
python generate.py --quantize true --prompt "Hello, my name is"
```

See `python generate.py --help` for more options.

&nbsp;

## Get involved!

We're in a quest towards fully open source AI, especially focusing on models in the 5-20B range, trained using the LLaMA approach (smaller models trained for longer).

<img align="right" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/Lit_LLaMA_Illustration3x.png" alt="Lit-LLaMA" width="128"/>

Join us and start contributing, especially on the following areas:

- [ ] Pre-training
- [ ] Fine-tuning (full and LoRA)
- [ ] Quantization
- [ ] Sparsification

Look at `train.py` for a starting point towards pre-training / fine-tuning using [Lightning Fabric](https://lightning.ai/docs/fabric/stable/).

Don't forget to [join our Discord](https://discord.gg/VptPCZkGNa)!

## Acknowledgements

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@FacebookResearch](https://github.com/facebookresearch) for the original [LLaMA implementation](https://github.com/facebookresearch/llama)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

## License

Lit-LLaMA is released under the [Apache 2.0](https://github.com/Lightning-AI/lightning-llama/blob/main/LICENSE) license.
