# NeurIPS 2023 LLM Efficiency Challenge Quickstart Guide

The [NeurIPS 2023 Efficiency Challenge](https://llm-efficiency-challenge.github.io/) is a competition focused on training **1 LLM for 24 hours on 1 GPU** – the team with the best LLM gets to present their results at NeurIPS 2023.

This quick start guide is a short starter guide illustrating the main steps to get started with Lit-GPT, which was selected as the competition's official starter kit.

&nbsp;

## Competition Facts

&nbsp;

**Permitted GPUs:**

- 1x A100 (40 GB RAM);
- 1x RTX 4090 (24 GB RAM).

&nbsp;

**Permitted models:**

- All transformer-based LLM base models that are not finetuned yet.

The subset of Lit-GPT models supported in this competition is listed in the table below.
These don't include models that have been finetuned or otherwise aligned, as per the rules of the challenge.

&nbsp;

| Models in Lit-GPT         | Reference                                                    |
| ------------------------- | ------------------------------------------------------------ |
| Meta AI Llama 2 Base      | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288)      |
| TII UAE Falcon Base       | [TII 2023](https://falconllm.tii.ae/)                        |
| OpenLM Research OpenLLaMA | [Geng & Liu 2023](https://github.com/openlm-research/open_llama) |
| EleutherAI Pythia         | [Biderman et al. 2023](https://arxiv.org/abs/2304.01373)     |
| StabilityAI StableLM Base | [Stability AI 2023](https://github.com/Stability-AI/StableLM) |

&nbsp;

**Permitted datasets**

Any open-source dataset is allowed. Originally, [per competition rules](https://llm-efficiency-challenge.github.io/challenge), datasets that utilize "generated content" from other LLMs were not permitted. However, the rules were recently softened to also allow LLM-generated datasets if those datasets are made available and if it is not against the usage restrictions and guidelines of the LLM. If you plan to use a specific dataset that is not explicitly listed on the [challenge website](https://llm-efficiency-challenge.github.io/challenge) or want to use LLM-generated data, it is recommended to reach out to the organizers and confirm that this is in line with the competition rules.

Examples of permitted datasets are the following:

- [Databricks-Dolly-15](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [OpenAssistant Conversations Dataset (oasst1)](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [The Flan Collection](https://github.com/google-research/FLAN/tree/main/flan/v2)

You are allowed to create your own datasets if they are made
publicly accessible under an open-source license, and they are not generated from other LLMs (even open-source ones).

Helpful competition rules relevant to the dataset choice:

- The maximum prompt/completion length the models are expected to handle is 2048 tokens.
- The evaluation will be on English texts only.

&nbsp;

**Submission deadline**

- October 25, 2023 ([Please check](https://llm-efficiency-challenge.github.io/dates) official website in case of updates.)

&nbsp;

## Lit-GPT Setup

Use the following steps to set up the Lit-GPT repository on your machine.

```shell
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
pip install -r requirements.txt tokenizers sentencepiece huggingface_hub
```

&nbsp;

## Downloading Model Checkpoints

This section explains how to download the StableLM 3B Base model, one of the smallest models supported in Lit-GPT (an even smaller, supported model is Pythia, which starts at 70M parameters). The downloaded and converted checkpoints will occupy approximately 28 Gb of disk space.

```bash
python scripts/download.py \
  --repo_id stabilityai/stablelm-base-alpha-3b

python scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

While StableLM 3B Base is useful as a first starter model to set things up, you may want to use the more capable Falcon 7B or Llama 2 7B/13B models later. See the [`download_*`](https://github.com/Lightning-AI/lit-gpt/tree/main/tutorials) tutorials in Lit-GPT to download other model checkpoints.

After downloading and converting the model checkpoint, you can test the model via the following command:

```bash
python generate/base.py \
  --prompt "LLM efficiency competitions are fun, because" \
  --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

&nbsp;

## Downloading and Preparing Datasets

The following command will download and preprocess the Dolly15k dataset for the StableLM 3B Base model:

```bash
python scripts/prepare_dolly.py \
  --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b \
  --destination_path data/dolly-stablelm3b
```

> [!NOTE]
> The preprocessed dataset is specific to the StableLM 3B model. If you use a different model like Falcon or Llama 2 later, you'll need to process the dataset with that model checkpoint directory. This is because each model uses a different tokenizer.

&nbsp;

## Finetuning

[Low-rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/) is a good choice for a first finetuning run. The Dolly dataset has ~15k samples, and the finetuning might take half an hour.

To accelerate this for testing purposes, edit the [./finetune/lora.py](https://github.com/Lightning-AI/lit-gpt/blob/main/finetune/lora.py) script and change `max_iters = 50000` to `max_iters = 500` at the top of the file.

> [!NOTE]
> The Dolly dataset has a relatively long context length, which could result in out-of-memory issues. The maximum context length that is used for the evaluation, [according to the official competition rules](https://llm-efficiency-challenge.github.io/question), is 2,048 tokens. Hence, it's highly recommended to prepare the dataset with a fixed max length, for example, `python scripts/prepare_dolly.py --max_seq_length 2048`.

The following command finetunes the model:

```bash
CUDA_VISIBLE_DEVICES=2 python finetune/lora.py \
  --data_dir data/dolly-stablelm3b \
  --checkpoint_dir "checkpoints/stabilityai/stablelm-base-alpha-3b" \
  --out_dir "out/stablelm3b/dolly/lora/experiment1" \
  --precision "bf16-true"
```

With 500 iterations, this takes approximately 1-2 min on an A100 and uses 26.30 GB GPU memory.

If you are using an RTX 4090, change `micro_batch_size=4` to `micro_batch_size=1` so that the model will only use 12.01 GB of memory.

(More finetuning settings are explained [here](https://lightning.ai/pages/community/tutorial/neurips2023-llm-efficiency-guide/#toc10).)

&nbsp;

## Local Evaluation

The official Lit-GPT competition will use a small subset of HELM tasks for model evaluation, which includes BigBench (general), MMLU (knowledge), TruthfulQA (knowledge and harm in a multiple choice format), CNN/DailyMail (news summarization), GSM8K (math), and BBQ (bias).

HELM is currently also being integrated into Lit-GPT to evaluate LLMs before submission.

However, a tool with a more convenient interface is Eleuther AI's [Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), which contains some tasks, for example, BigBench, TruthfulQA, and GSM8k, that overlap with HELM. We can set up the Evaluation Harness as follows:

```bash
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@master
```

And then we can use it via the following command:

```bash
python eval/lm_eval_harness.py \
  --checkpoint_dir "checkpoints/stabilityai/stablelm-base-alpha-3b" \
  --precision "bf16-true" \
  --eval_tasks "[truthfulqa_mc,gsm8k]" \
  --save_filepath "results-stablelm-3b.json"
```

(You can find a full task list in the task table [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md).)

To evaluate a LoRA-finetuned model, you need to first merge the LoRA weights with the base model to create a new checkpoint file:

```bash
python scripts/merge_lora.py \
  --checkpoint_dir "checkpoints/stabilityai/stablelm-base-alpha-3b/" \
  --lora_path "out/stablelm3b/dolly/lora/experiment1/lit_model_lora_finetuned.pth" \
  --out_dir "out/lora_merged/stablelm-base-alpha-3b/"
```

```bash
cp checkpoints/stabilityai/stablelm-base-alpha-3b/*.json \
out/lora_merged/stablelm-base-alpha-3b/
```

For more information on LoRA weight merging, please see the
[Merging LoRA Weights](finetune_lora.md#merging-lora-weights)
section of the LoRA finetuning documentation.

After merging the weights, we can use the `lm_eval_harness.py` similar to before with the only difference that we now use the new
checkpoint folder containing the merged LoRA model:

```bash
python eval/lm_eval_harness.py \
  --checkpoint_dir "out/lora_merged/stablelm-base-alpha-3b" \
  --precision "bf16-true" \
  --eval_tasks "[truthfulqa_mc,gsm8k]" \
  --save_filepath "results-stablelm-3b.json"
```

&nbsp;

## Submission

You will be required to submit a Docker image for the submission itself. Fortunately, the organizers have a GitHub repository with the exact steps [here](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge) and a toy-submission setup guide to test your model locally before submission.

&nbsp;

## Additional Information & Resources

- [The official NeurIPS 2023 LLM Efficiency Challenge competition website](https://llm-efficiency-challenge.github.io/)
- A more extensive guide, including environment setup tips: [The NeurIPS 2023 LLM Efficiency Challenge Starter Guide](https://lightning.ai/pages/community/tutorial/neurips2023-llm-efficiency-guide)
- [Official competition Discord](https://discord.com/login?redirect_to=%2Fchannels%2F1077906959069626439%2F1134560480795570186) and [Lightning AI + Lit-GPT Discord](https://discord.com/invite/MWAEvnC5fU)
- LoRA vs Adapter vs Adapter v2 comparison in Lit-GPT using Falcon 7B: [Finetuning Falcon LLMs More Efficiently With LoRA and Adapters](https://lightning.ai/pages/community/finetuning-falcon-efficiently/)
- [Dealing with out-of-memory (OOM) errors in Lit-GPT](oom.md)
- Introduction to Fabric (an API to access more advanced PyTorch features used in Lit-GPT) and memory saving tips: [Optimizing Memory Usage for Training LLMs and Vision Transformers in PyTorch](https://lightning.ai/pages/community/tutorial/pytorch-memory-vit-llm/)
