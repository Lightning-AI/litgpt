<div align="center">


# ⚡ LitGPT

**涵盖 20+ 款高性能大语言模型（LLM），提供开箱即用的工业级预训练、微调与大规模部署配方。**

<pre>
✅ 从零构建的纯粹实现          ✅ 零抽象层封装               ✅ 新手友好
✅ Flash Attention 高速注意力   ✅ FSDP 完全分片数据并行       ✅ LoRA、QLoRA、Adapter
✅ 显存极致优化 (fp4/8/16/32)   ✅ 支持 1 至 1000+ GPU/TPU    ✅ 覆盖 20+ 款主流大模型
</pre>


---


![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)
![cpu-tests](https://github.com/Lightning-AI/litgpt/actions/workflows/cpu-tests.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE.md) [![Discord](https://img.shields.io/discord/1077906959069626439)](https://discord.gg/VptPCZkGNa)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <a href="#快速上手">快速上手</a> •
  <a href="#20-款主流大模型任选">支持模型</a> •
  <a href="#微调大语言模型-finetune">微调模型</a> •
  <a href="#部署大语言模型-deploy">服务部署</a> •
  <a href="#所有工作流概览">全部工作流</a> •
  <a href="#前沿领先特性">核心特性</a> •
  <a href="#训练配方-recipes">训练配方 (YAML)</a> •
  <a href="https://lightning.ai/">Lightning AI</a> •
  <a href="#实用教程">实战教程</a>
</p>

&nbsp;

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-quick-start">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="快速开始"/>
</a>

&nbsp;

</div>

# 正在寻找云端算力？
已有超过 340,000 名开发者选择专为 PyTorch 和 PyTorch Lightning 打造的 [Lightning Cloud](https://lightning.ai/?utm_source=litgpt_readme&utm_medium=referral&utm_campaign=litgpt_readme)：
- [GPU 算力](https://lightning.ai/pricing?utm_source=litgpt_readme&utm_medium=referral&utm_campaign=litgpt_readme)：低至每小时 $0.19 起。
- [算力集群 (Clusters)](https://lightning.ai/clusters?utm_source=litgpt_readme&utm_medium=referral&utm_campaign=litgpt_readme)：前沿尖端的大规模训练与推理集群。
- [AI Studio (氛围训练 vibe train)](https://lightning.ai/studios?utm_source=litgpt_readme&utm_medium=referral&utm_campaign=litgpt_readme)：配备 AI 辅助调试、调优及训练的一站式云端工作区。
- [AI Studio (氛围部署 vibe deploy)](https://lightning.ai/studios?utm_source=litgpt_readme&utm_medium=referral&utm_campaign=litgpt_readme)：配备 AI 辅助模型优化与生产级部署的协作空间。
- [云端 Notebook](https://lightning.ai/notebooks?utm_source=litgpt_readme&utm_medium=referral&utm_campaign=litgpt_readme)：持久化 GPU 开发环境，结合 AI 编码与分析助手。
- [模型推理 (Inference)](https://lightning.ai/deploy?utm_source=litgpt_readme&utm_medium=referral&utm_campaign=litgpt_readme)：将模型一键发布为高性能推理 API。

# 极速微调、预训练与大模型推理 ⚡⚡
库中的每一个 LLM 均采用从零实现的单文件代码架构，**无多余抽象封装**，赋予开发者**完全的控制力**；不仅运行速度极致飞快、代码极简纯粹，且在企业级规模下展现卓越性能。

✅ **企业级就绪 -** Apache 2.0 商业开源许可，支持无限制的企业级集成与商用。</br>
✅ **开发者友好 -** 单文件实现、无重重内部嵌套抽象，断点单步调试直观清晰。</br>
✅ **性能极致调优 -** 架构专为最大化硬件吞吐、降低训练成本并加快收敛设计。</br>
✅ **成熟严谨配方 -** 经过企业级真实生产场景深度验证的高效预训练与微调配方。</br>

&nbsp;

# 快速上手
安装 LitGPT：
```bash
pip install 'litgpt[extra]'
```

加载并直接调用 [20+ 款大模型](#20-款主流大模型任选)中的任意一款：
```python
from litgpt import LLM

llm = LLM.load("microsoft/phi-2")
text = llm.generate("Fix the spelling: Every fall, the family goes to the mountains.")
print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.
```

&nbsp;

✅ 针对极速推理全面优化</br>
✅ 原生集成量化支持</br>
✅ 轻松运行于低显存消费级 GPU</br>
✅ 毫无冗余复杂的内部抽象黑盒</br>
✅ 专为生产级应用环境量身打造</br>

<details>
  <summary>进阶安装选项</summary>

从源码编译安装：

```bash
git clone https://github.com/Lightning-AI/litgpt
cd litgpt
# 若使用 uv
uv sync --all-extras
# 若使用 pip
pip install -e ".[extra,compiler,test]"
```
</details>

[查阅完整的 Python API 文档](tutorials/python-api.md)。

&nbsp;

---
# 20+ 款主流大模型任选
所有模型结构均从零编写，去除了复杂的封装层以追求极限运行性能：

| 模型系列 | 模型参数量 | 开源团队 | 引用文献 |
|----|----|----|----|
| Llama 3, 3.1, 3.2, 3.3 | 1B, 3B, 8B, 70B, 405B | Meta AI | [Meta AI 2024](https://github.com/meta-llama/llama3) |
| Code Llama | 7B, 13B, 34B, 70B | Meta AI | [Rozière et al. 2023](https://arxiv.org/abs/2308.12950) |
| CodeGemma | 7B | Google | [Google Team, Google Deepmind](https://ai.google.dev/gemma/docs/codegemma) |
| Gemma 2 | 2B, 9B, 27B | Google | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) |
| Phi 4 | 14B | Microsoft Research | [Abdin et al. 2024](https://arxiv.org/abs/2412.08905) |
| Qwen2.5 | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | 阿里巴巴 (Alibaba Group) | [Qwen Team 2024](https://qwenlm.github.io/blog/qwen2.5/) |
| Qwen2.5 Coder | 0.5B, 1.5B, 3B, 7B, 14B, 32B | 阿里巴巴 (Alibaba Group) | [Hui, Binyuan et al. 2024](https://arxiv.org/abs/2409.12186) |
| R1 Distill Llama | 8B, 70B | 深度求索 (DeepSeek AI) | [DeepSeek AI 2025](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) |
| ... | ... | ... | ... |

<details>
  <summary>点击展开 20+ 款模型完整列表</summary>

&nbsp;

#### 全部支持模型

| 模型系列 | 模型参数量 | 开源团队 | 引用文献 |
|----|----|----|----|
| CodeGemma | 7B | Google | [Google Team, Google Deepmind](https://ai.google.dev/gemma/docs/codegemma) |
| Code Llama | 7B, 13B, 34B, 70B | Meta AI | [Rozière et al. 2023](https://arxiv.org/abs/2308.12950) |
| Falcon | 7B, 40B, 180B | TII UAE | [TII 2023](https://falconllm.tii.ae) |
| Falcon 3 | 1B, 3B, 7B, 10B | TII UAE | [TII 2024](https://huggingface.co/blog/falcon3) |
| FreeWilly2 (Stable Beluga 2) | 70B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stable-beluga-large-instruction-fine-tuned-models) |
| Function Calling Llama 2 | 7B | Trelis | [Trelis et al. 2023](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2) |
| Gemma | 2B, 7B | Google | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf) |
| Gemma 2 | 9B, 27B | Google | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) |
| Gemma 3 | 1B, 4B, 12B, 27B | Google | [Google Team, Google Deepmind](https://arxiv.org/pdf/2503.19786) |
| Llama 2 | 7B, 13B, 70B | Meta AI | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288) |
| Llama 3.1 | 8B, 70B | Meta AI | [Meta AI 2024](https://github.com/meta-llama/llama3) |
| Llama 3.2 | 1B, 3B | Meta AI | [Meta AI 2024](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) |
| Llama 3.3 | 70B | Meta AI | [Meta AI 2024](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| Mathstral | 7B | Mistral AI | [Mistral AI 2024](https://mistral.ai/news/mathstral/) |
| MicroLlama | 300M | Ken Wang | [MicroLlama 代码库](https://github.com/keeeeenw/MicroLlama) |
| Mixtral MoE | 8x7B | Mistral AI | [Mistral AI 2023](https://mistral.ai/news/mixtral-of-experts/) |
| Mistral | 7B, 123B | Mistral AI | [Mistral AI 2023](https://mistral.ai/news/announcing-mistral-7b/) |
| Mixtral MoE | 8x22B | Mistral AI | [Mistral AI 2024](https://mistral.ai/news/mixtral-8x22b/) |
| OLMo | 1B, 7B | 艾伦人工智能研究所 (AI2) | [Groeneveld et al. 2024](https://aclanthology.org/2024.acl-long.841/) |
| OpenLLaMA | 3B, 7B, 13B | OpenLM Research | [Geng & Liu 2023](https://github.com/openlm-research/open_llama) |
| Phi 1.5 & 2 | 1.3B, 2.7B | Microsoft Research | [Li et al. 2023](https://arxiv.org/abs/2309.05463) |
| Phi 3 | 3.8B | Microsoft Research | [Abdin et al. 2024](https://arxiv.org/abs/2404.14219) |
| Phi 4 | 14B | Microsoft Research | [Abdin et al. 2024](https://arxiv.org/abs/2412.08905) |
| Phi 4 Mini Instruct | 3.8B | Microsoft Research | [Microsoft 2025](https://arxiv.org/abs/2503.01743) |
| Phi 4 Mini Reasoning | 3.8B | Microsoft Research | [Xu, Peng et al. 2025](https://arxiv.org/abs/2504.21233) |
| Phi 4 Reasoning | 3.8B | Microsoft Research | [Abdin et al. 2025](https://arxiv.org/abs/2504.21318) |
| Phi 4 Reasoning Plus | 3.8B | Microsoft Research | [Abdin et al. 2025](https://arxiv.org/abs/2504.21318) |
| Platypus | 7B, 13B, 70B | Lee et al. | [Lee, Hunter, and Ruiz 2023](https://arxiv.org/abs/2308.07317) |
| Pythia | {14,31,70,160,410}M, {1,1.4,2.8,6.9,12}B | EleutherAI | [Biderman et al. 2023](https://arxiv.org/abs/2304.01373) |
| Qwen2.5 | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | 阿里巴巴 (Alibaba Group) | [Qwen Team 2024](https://qwenlm.github.io/blog/qwen2.5/) |
| Qwen2.5 Coder | 0.5B, 1.5B, 3B, 7B, 14B, 32B | 阿里巴巴 (Alibaba Group) | [Hui, Binyuan et al. 2024](https://arxiv.org/abs/2409.12186) |
| Qwen2.5 1M (超长上下文) | 7B, 14B | 阿里巴巴 (Alibaba Group) | [Qwen Team 2025](https://qwenlm.github.io/blog/qwen2.5-1m/) |
| Qwen2.5 Math | 1.5B, 7B, 72B | 阿里巴巴 (Alibaba Group) | [An, Yang et al. 2024](https://arxiv.org/abs/2409.12122) |
| QwQ | 32B | 阿里巴巴 (Alibaba Group) | [Qwen Team 2025](https://qwenlm.github.io/blog/qwq-32b/) |
| QwQ-Preview | 32B | 阿里巴巴 (Alibaba Group) | [Qwen Team 2024](https://qwenlm.github.io/blog/qwq-32b-preview/) |
| Qwen3 | 0.6B, 1.7B, 4B{Hybrid, Thinking-2507, Instruct-2507}, 8B, 14B, 32B | 阿里巴巴 (Alibaba Group) | [Qwen Team 2025](https://arxiv.org/abs/2505.09388/) |
| Qwen3 MoE | 30B{Hybrid, Thinking-2507, Instruct-2507}, 235B{Hybrid, Thinking-2507, Instruct-2507} | 阿里巴巴 (Alibaba Group) | [Qwen Team 2025](https://arxiv.org/abs/2505.09388/) |
| R1 Distill Llama | 8B, 70B | 深度求索 (DeepSeek AI) | [DeepSeek AI 2025](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) |
| SmolLM2 | 135M, 360M, 1.7B | Hugging Face | [Hugging Face 2024](https://github.com/huggingface/smollm) |
| Salamandra | 2B, 7B | 巴塞罗那超级计算中心 (BSC) | [BSC-LTC 2024](https://github.com/langtech-bsc/salamandra) |
| StableCode | 3B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding) |
| StableLM | 3B, 7B | Stability AI | [Stability AI 2023](https://github.com/Stability-AI/StableLM) |
| StableLM Zephyr | 3B | Stability AI | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding) |
| TinyLlama | 1.1B | Zhang et al. | [Zhang et al. 2023](https://github.com/jzhang38/TinyLlama) |

**提示**：运行 `litgpt download list` 命令行即可列出所有当前可用的完整模型列表。

</details>

&nbsp;

---

# 工作流指南

<p align="center">
  <a href="#微调大语言模型-finetune">微调模型</a> •
  <a href="#预训练大语言模型-pretrain">预训练模型</a> •
  <a href="#继续预训练大语言模型-continue-pretraining">继续预训练</a> •
  <a href="#评测大语言模型-evaluate">评估评测</a> •
  <a href="#部署大语言模型-deploy">服务部署</a> •
  <a href="#测试与交互-chat">测试交互</a>
</p>

&nbsp;

直接使用命令行工具（CLI），在您自己的私有数据上轻松运行预训练或微调等高阶工作流。

## 所有工作流概览
安装 LitGPT 后，只需指定目标模型和要运行的工作流动作（微调、预训练、评测、部署等）：

```bash
# litgpt [action] [model]
litgpt  serve     meta-llama/Llama-3.2-3B-Instruct
litgpt  finetune  meta-llama/Llama-3.2-3B-Instruct
litgpt  pretrain  meta-llama/Llama-3.2-3B-Instruct
litgpt  chat      meta-llama/Llama-3.2-3B-Instruct
litgpt  evaluate  meta-llama/Llama-3.2-3B-Instruct
```

&nbsp;

----

## 微调大语言模型 (Finetune)

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-finetune">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/run-on-studio.svg" height="36px" alt="在 Studio 上运行"/>
</a>
</div>

&nbsp;

微调是指以预训练模型为基座，在更小、更专业的下游任务数据集上进行进阶训练，使其针对特定任务或业务场景深度特化的技术过程。

&nbsp;

```bash
# 0) 准备自定义数据集
curl -L https://huggingface.co/datasets/ksaw008/finance_alpaca/resolve/main/finance_alpaca.json -o my_custom_dataset.json

# 1) 微调模型（自动下载权重）
litgpt finetune microsoft/phi-2   --data JSON   --data.json_path my_custom_dataset.json   --data.val_split_fraction 0.1   --out_dir out/custom-model

# 2) 交互测试微调后的模型
litgpt chat out/custom-model/final

# 3) 部署微调后的模型服务
litgpt serve out/custom-model/final
```

[查阅完整的模型微调文档](tutorials/finetune.md)。

&nbsp;

----

## 部署大语言模型 (Deploy)

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-serve">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" height="36px" alt="在 Studio 上部署"/>
</a>
</div>

&nbsp;

将预训练或微调完成的大模型部署上线，以供实际业务应用调用。Deploy 模块会自动启动轻量级高并发 Web 服务，网站或应用程序可随时通过 HTTP API 访问。

```bash
# 部署开箱即用的预训练模型
litgpt serve microsoft/phi-2

# 部署自己微调的模型权重
litgpt serve path/to/microsoft/phi-2/checkpoint
```

<details>
  <summary>查看通过 Python 客户端请求服务器的代码：</summary>

&nbsp;

在另一个终端或代码中测试 API 服务，并将其集成到你的 AI 应用产品中：
```python
# 3) 请求模型服务（在独立的 Python 会话中）
import requests, json
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"prompt": "Fix typos in the following sentence: Example input"}
)
print(response.json()["output"])
```
</details>

[查阅完整的模型部署文档](tutorials/deploy.md)。

&nbsp;

----

## 评测大语言模型 (Evaluate)
评测大语言模型在多项基准测试上的表现，直观衡量其文本理解与生成质量。简而言之，评测可以验证模型在大学阶段化学、编程、常识推理等任务上的解题水平（如 MMLU、Truthful QA 等基准）：

```bash
litgpt evaluate microsoft/phi-2 --tasks 'truthfulqa_mc2,mmlu'
```

[查阅完整的模型评测文档](tutorials/evaluation.md)。

&nbsp;

----

## 测试与交互 (Chat)

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-chat">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/run-on-studio.svg" height="36px" alt="在 Studio 上运行"/>
</a>
</div>

&nbsp;

通过交互式聊天终端直观检验模型的生成效果。使用 `chat` 命令即可进行多轮对话、抽取向量嵌入（Embeddings）等：

以 Phi-2 模型的调用为例：
```bash
litgpt chat microsoft/phi-2

>> Prompt: What do Llamas eat?
```

<details>
  <summary>完整操作代码：</summary>

&nbsp;

```bash
# 1) 列出所有支持的预置大模型
litgpt download list

# 2) 运行模型（自动拉取权重）
litgpt chat microsoft/phi-2

>> Prompt: What do Llamas eat?
```

部分受控商业模型的下载需要提供授权 Access Token。详情请参考[模型下载文档](tutorials/download_model_weights.md#specific-models-and-access-tokens)。

</details>

[查阅完整的推理与对话文档](tutorials/inference.md)。

&nbsp;

----

## 预训练大语言模型 (Pretrain)

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-pretrain">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/run-on-studio.svg" height="36px" alt="在 Studio 上运行"/>
</a>
</div>

&nbsp;

预训练是指在模型针对特定任务微调之前，让其在海量无标注文本语料中通过自监督学习掌握语言表征与通用知识的基础训练过程。

<details>
  <summary>查看操作代码：</summary>

&nbsp;

```bash
mkdir -p custom_texts
curl https://www.gutenberg.org/cache/epub/24440/pg24440.txt --output custom_texts/book1.txt
curl https://www.gutenberg.org/cache/epub/26393/pg26393.txt --output custom_texts/book2.txt

# 1) 下载分词器 (Tokenizer)
litgpt download EleutherAI/pythia-160m   --tokenizer_only True

# 2) 预训练模型
litgpt pretrain EleutherAI/pythia-160m   --tokenizer_dir EleutherAI/pythia-160m   --data TextFiles   --data.train_data_path "custom_texts/"   --train.max_tokens 10_000_000   --out_dir out/custom-model

# 3) 交互测试模型
litgpt chat out/custom-model/final
```
</details>

[查阅完整的模型预训练文档](tutorials/pretrain.md)。

&nbsp;

----

## 继续预训练大语言模型 (Continue Pretraining)

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litgpt-continue-pretraining">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/run-on-studio.svg" height="36px" alt="在 Studio 上运行"/>
</a>
</div>

&nbsp;

继续预训练（领域适配）是另一种高效的模型特化手段，它直接基于已有的预训练模型权重，在垂直领域语料上继续进行自监督训练：

<details>
  <summary>查看操作代码：</summary>

&nbsp;

```bash
mkdir -p custom_texts
curl https://www.gutenberg.org/cache/epub/24440/pg24440.txt --output custom_texts/book1.txt
curl https://www.gutenberg.org/cache/epub/26393/pg26393.txt --output custom_texts/book2.txt

# 1) 继续预训练模型（自动拉取基座权重）
litgpt pretrain EleutherAI/pythia-160m   --tokenizer_dir EleutherAI/pythia-160m   --initial_checkpoint_dir EleutherAI/pythia-160m   --data TextFiles   --data.train_data_path "custom_texts/"   --train.max_tokens 10_000_000   --out_dir out/custom-model

# 2) 测试训练完成的模型
litgpt chat out/custom-model/final
```

</details>

[查阅完整的继续预训练文档](tutorials/pretrain.md#continued-pretraining-on-custom-data)。

&nbsp;

----

# 前沿领先特性

✅ 前沿性能加速优化：Flash Attention v2、基于完全分片数据并行（FSDP）的多卡并行训练、[可选的 CPU 内存卸载 (Offloading)](tutorials/oom.md#do-sharding-across-multiple-gpus)，以及 [云端 TPU 与 XLA 原生支持](extensions/xla)。</br>
✅ 一站式覆盖 [预训练 (Pretrain)](tutorials/pretrain.md)、[微调 (Finetune)](tutorials/finetune.md) 与 [服务部署 (Deploy)](tutorials/inference.md)。</br>
✅ 低精度算力节省：支持 FP16、BF16 以及 FP16/FP32 混合精度训练。</br>
✅ 极致降低显存占用：原生提供 [模型量化 (Quantization)](tutorials/quantize.md)，涵盖 4 位浮点（fp4）、8 位整数（int8）及双重量化（double quantization）。</br>
✅ 丰富的 [预置配置文件库](config_hub)，开箱即达最佳运行性能。</br>
✅ 主流参数高效微调（PEFT）技术全覆盖：[LoRA](tutorials/finetune_lora.md)、[QLoRA](tutorials/finetune_lora.md)、[Adapter](tutorials/finetune_adapter.md) 及 [Adapter v2](tutorials/finetune_adapter.md)。</br>
✅ 支持将模型权重 [无缝导出转换](tutorials/convert_lit_models.md) 为其他主流格式。</br>
✅ 内置丰富的大众开源数据集，支持 [预训练](tutorials/pretrain.md) 与 [微调](tutorials/prepare_dataset.md)，并提供灵活的 [自定义私有数据集适配指南](tutorials/prepare_dataset.md#preparing-custom-datasets-for-instruction-finetuning)。</br>
✅ 纯粹、易读、易改的代码结构，方便快速验证前沿科研想法。</br>

&nbsp;

---

# 训练配方 (Recipes)

LitGPT 自带经过充分实验验证的预置配方（YAML 配置文件），可直接在各类硬件与约束条件下训练模型。这些配方参数均基于我们在大量实验中表现最优的超参数精选而成。

浏览全部预置训练配方请前往 [此处](config_hub)。

### 使用范例

```bash
litgpt finetune   --config https://raw.githubusercontent.com/Lightning-AI/litgpt/main/config_hub/finetune/llama-2-7b/lora.yaml
```
<details>
  <summary>✅ 使用配置文件按需深度定制训练</summary>

配置文件支持细粒度定制所有训练参数，例如：

```yaml
# 用于微调加载的基座模型权重目录路径 (类型: Path, 默认: checkpoints/stabilityai/stablelm-base-alpha-3b)
checkpoint_dir: checkpoints/meta-llama/Llama-2-7b-hf

# 权重检查点与训练日志保存目录 (类型: Path, 默认: out/lora)
out_dir: out/finetune/qlora-llama2-7b

# 微调所采用的计算精度。可选: "bf16-true", "bf16-mixed", "32-true" (类型: Optional[str], 默认: null)
precision: bf16-true

...
```
</details>

<details>
  <summary>✅ 范例：LoRA 微调 YAML 配置</summary>

&nbsp;

```yaml
# 用于微调加载的基座模型权重目录路径 (类型: Path, 默认: checkpoints/stabilityai/stablelm-base-alpha-3b)
checkpoint_dir: checkpoints/meta-llama/Llama-2-7b-hf

# 权重检查点与训练日志保存目录 (类型: Path, 默认: out/lora)
out_dir: out/finetune/qlora-llama2-7b

# 微调所采用的计算精度。可选: "bf16-true", "bf16-mixed", "32-true" (类型: Optional[str], 默认: null)
precision: bf16-true

# 是否使用特定算法对模型进行量化。更多信息见 ``tutorials/quantize.md`` (可选: 'nf4', 'nf4-dq', 'fp4', 'fp4-dq', 'int8-training', 默认: null)
quantize: bnb.nf4

# 使用的加速设备/GPU数量 (类型: Union[int, str], 默认: 1)
devices: 1

# 使用的计算节点数 (类型: int, 默认: 1)
num_nodes: 1

# LoRA 秩 (Rank) (类型: int, 默认: 8)
lora_r: 32

# LoRA 缩放系数 (Alpha) (类型: int, 默认: 16)
lora_alpha: 16

# LoRA Dropout 比例 (类型: float, 默认: 0.05)
lora_dropout: 0.05

# 是否在 Attention 的 Query 权重上应用 LoRA (类型: bool, 默认: True)
lora_query: true

# 是否在 Attention 的 Key 权重上应用 LoRA (类型: bool, 默认: False)
lora_key: false

# 是否在 Attention 的 Value 权重上应用 LoRA (类型: bool, 默认: True)
lora_value: true

# 是否在 Attention 模块的输出投影矩阵上应用 LoRA (类型: bool, 默认: False)
lora_projection: false

# 是否在 Attention 模块的 MLP 前馈网络上应用 LoRA (类型: bool, 默认: False)
lora_mlp: false

# 是否在 GPT 输出头上应用 LoRA (类型: bool, 默认: False)
lora_head: false

# 数据集相关参数。若未提供，默认为 ``litgpt.data.Alpaca``
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

# 训练相关超参数。详细定义见 ``litgpt.args.TrainArgs``
train:

  # 保存权重检查点之间的优化器步数 (类型: Optional[int], 默认: 1000)
  save_interval: 200

  # 记录训练指标之间的迭代次数 (类型: int, 默认: 1)
  log_interval: 1

  # 各数据并行 Rank 间每次优化器更新所处理的总样本数 (全局 Batch Size) (类型: int, 默认: 128)
  global_batch_size: 8

  # 每个数据并行 Rank 单步处理的样本数 (Micro Batch Size) (类型: int, 默认: 4)
  micro_batch_size: 2

  # 学习率预热步数 (Warmup) (类型: int, 默认: 100)
  lr_warmup_steps: 10

  # 训练 Epoch 总轮数 (类型: Optional[int], 默认: 5)
  epochs: 4

  # 训练的最大 Token 数量 (类型: Optional[int], 默认: null)
  max_tokens:

  # 运行的最大优化器步数上限 (类型: Optional[int], 默认: null)
  max_steps:

  # 样本序列最大长度限制 (类型: Optional[int], 默认: null)
  max_seq_length: 512

  # 是否将词嵌入权重与语言模型输出头权重绑定 (Tie Embeddings) (类型: Optional[bool], 默认: null)
  tie_embeddings:

  # 学习率 (类型: float, 默认: 0.0003)
  learning_rate: 0.0002

  # 权重衰减 (Weight Decay) (类型: float, 默认: 0.02)
  weight_decay: 0.0

  # Adam 优化器 beta1 参数 (类型: float, 默认: 0.9)
  beta1: 0.9

  # Adam 优化器 beta2 参数 (类型: float, 默认: 0.95)
  beta2: 0.95

  # 梯度裁剪最大范数 (类型: Optional[float], 默认: null)
  max_norm:

  # 最小学习率下限 (类型: float, 默认: 6e-05)
  min_lr: 6.0e-05

# 评测相关参数。详细定义见 ``litgpt.args.EvalArgs``
eval:

  # 评测之间的优化器步数间隔 (类型: int, 默认: 100)
  interval: 100

  # 评估生成的最长 Token 数量 (类型: Optional[int], 默认: 100)
  max_new_tokens: 100

  # 评估最大迭代轮数 (类型: int, 默认: 100)
  max_iters: 100

# 用于记录训练指标的日志记录器类型 (可选: 'wandb', 'tensorboard', 'csv', 默认: csv)
logger_name: csv

# 保证实验可复现性的随机种子 (类型: int, 默认: 1337)
seed: 1337
```
</details>

<details>
  <summary>✅ 通过命令行动态覆盖任意配置参数：</summary>

```bash
litgpt finetune   --config https://raw.githubusercontent.com/Lightning-AI/litgpt/main/config_hub/finetune/llama-2-7b/lora.yaml   --lora_r 4
```
</details>

&nbsp;

----

# 代表性项目与落地成果

LitGPT 成功赋能了许多顶尖的 AI 科研项目、行业倡议、算法挑战赛以及企业落地。若您希望在此展示您的代表性项目，欢迎提交 Pull Request！

<details>
  <summary>📊 SAMBA：面向无限上下文高效建模的混合状态空间模型</summary>

由微软研究团队主导的 [Samba](https://github.com/microsoft/Samba) 项目构建于 LitGPT 代码库之上，通过创新性地将状态空间模型（SSM）与滑动窗口注意力机制相结合，显著超越了纯状态空间模型的表达能力。

</details>

<details>
  <summary>🏆 NeurIPS 2023 大语言模型能效挑战赛：1 个 LLM + 1 张 GPU + 1 天</summary>

LitGPT 官方仓库受选为 [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io) 的官方官方基线启动套件。该赛事旨在探索仅使用单卡 GPU 在 24 小时内完成非指令微调基础大模型的高效微调方案。

</details>

<details>
  <summary>🦙 TinyLlama：卓越开源小语言模型</summary>

LitGPT 全程赋能了知名开源项目 [TinyLlama](https://github.com/jzhang38/TinyLlama) 及其研究论文 [《TinyLlama: An Open-Source Small Language Model》](https://arxiv.org/abs/2401.02385)。

</details>

<details>
  <summary>🍪 MicroLlama：MicroLlama-300M</summary>

[MicroLlama](https://github.com/keeeeenw/MicroLlama) 是一个基于 500 亿 Token 数据从零预训练完成的 300M 参数量超轻量 Llama 架构模型，由 TinyLlama 与 LitGPT 共同驱动。
</details>

<details>
  <summary>🔬 以更少 Token 高效预训练小型基座语言模型</summary>

学术论文 [《Pre-training Small Base LMs with Fewer Tokens》](https://arxiv.org/abs/2404.08634) 借助 LitGPT，通过继承大模型部分 Transformer 模块并在极小比例的数据量上进行轻量预训练，证实了小型模型在耗费极低数据与算力资源的前提下，依然能够媲美大模型的强劲表现。

</details>

&nbsp;

----

# 生态社区

我们热忱欢迎所有开源贡献者，无论您的经验多寡或硬件设备条件如何。您的每一份贡献都极其宝贵，期待与您在开放包容的社区氛围中共同成长！

- [提交功能建议或反馈 Bug](https://github.com/Lightning-AI/litgpt/issues)
- [开启你的首次开源贡献指南](https://lightning.ai/pages/community/tutorial/how-to-contribute-to-litgpt/)
- [加入官方 Discord 开发者社区](https://discord.gg/VptPCZkGNa)

&nbsp;

# 实用教程

🚀 [从零上手 LitGPT](tutorials/0_to_litgpt.md)</br>
⚡️ [模型微调教程（涵盖 LoRA、QLoRA 及 Adapter）](tutorials/finetune.md)</br>
🤖 [从零预训练指南](tutorials/pretrain.md)</br>
💬 [大模型效果评测指南](tutorials/evaluation.md)</br>
📘 [支持数据集与私有数据集准备](tutorials/prepare_dataset.md)</br>
🧹 [模型量化实战](tutorials/quantize.md)</br>
🤯 [显存溢出 (OOM) 疑难排查与应对秘籍](tutorials/oom.md)</br>
🧑🏽‍💻 [在云端 TPU 上进行加速训练](extensions/xla)</br>

&nbsp;

----

### 致谢

本项目在实现上承袭扩展了 [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) 与 [nanoGPT](https://github.com/karpathy/nanoGPT)，底层**由 [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡ 强力驱动**。

- [@karpathy](https://github.com/karpathy) 贡献的 [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) 贡献的 [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) 及 [评测工具集 Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [@TimDettmers](https://github.com/TimDettmers) 贡献的 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 低显存库
- [@Microsoft](https://github.com/microsoft) 贡献的 [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) 贡献的 [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

### 开源许可

LitGPT 基于 [Apache 2.0](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE.md) 开源协议分发。

### 项目引用

若您在科研或生产项目中使用了 LitGPT，请参考如下格式进行引用：

```bibtex
@misc{litgpt-2023,
  author       = {Lightning AI},
  title        = {LitGPT},
  howpublished = {\url{https://github.com/Lightning-AI/litgpt}},
  year         = {2023},
}
```

&nbsp;

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
