# Download Model Weights with LitGPT

LitGPT supports a variety of LLM architectures with publicly available weights. You can download model weights and access a list of supported models using the LitGPT `download.py` script.


| Model                                        | Model size                               | Reference                                                                                                                    |
|----------------------------------------------|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Code Llama by Meta AI                        | 7B, 13B, 34B, 70B                        | [Rozière et al. 2023](https://arxiv.org/abs/2308.12950)                                                                      |
| Dolly by Databricks                          | 3B, 7B, 12B                              | [Conover et al. 2023](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) |
| Falcon by TII UAE                            | 7B, 40B, 180B                            | [TII 2023](https://falconllm.tii.ae)                                                                                         |
| FreeWilly2 (Stable Beluga 2) by Stability AI | 70B                                      | [Stability AI 2023](https://stability.ai/blog/stable-beluga-large-instruction-fine-tuned-models)                             |
| Function Calling Llama 2 by Trelis           | 7B                                       | [Trelis et al. 2023](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2)                                   |
| Gemma by Google                              | 2B, 7B                                   | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)                         |
| Llama 2 by Meta AI                           | 7B, 13B, 70B                             | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288)                                                                      |
| LongChat by LMSYS                            | 7B, 13B                                  | [LongChat Team 2023](https://lmsys.org/blog/2023-06-29-longchat/)                                                            |
| Mistral and Mixtral by Mistral AI            | 7B                                       | [Mistral website](https://mistral.ai/)                                                                                       |
| Nous-Hermes by NousResearch                  | 7B, 13B, 70B                             | [Org page](https://huggingface.co/NousResearch)                                                                              |
| OpenLLaMA by OpenLM Research                 | 3B, 7B, 13B                              | [Geng & Liu 2023](https://github.com/openlm-research/open_llama)                                                             |
| Phi by Microsoft Research                    | 1.3B, 2.7B                               | [Li et al. 2023](https://arxiv.org/abs/2309.05463)                                                                           |
| Platypus by Lee at el.                       | 7B, 13B, 70B                             | [Lee, Hunter, and Ruiz 2023](https://arxiv.org/abs/2308.07317)                                                               |
| Pythia by EleutherAI                         | {14,31,70,160,410}M, {1,1.4,2.8,6.9,12}B | [Biderman et al. 2023](https://arxiv.org/abs/2304.01373)                                                                     |
| RedPajama-INCITE by Together                 | 3B, 7B                                   | [Together 2023](https://together.ai/blog/redpajama-models-v1)                                                                |
| StableCode by Stability AI                   | 3B                                       | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)                                           |
| StableLM by Stability AI                     | 3B, 7B                                   | [Stability AI 2023](https://github.com/Stability-AI/StableLM)                                                                |
| StableLM Zephyr by Stability AI              | 3B                                       | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)                                           |
| TinyLlama by Zhang et al.                    | 1.1B                                     | [Zhang et al. 2023](https://github.com/jzhang38/TinyLlama)                                                                   |
| Vicuna by LMSYS                              | 7B, 13B, 33B                             | [Li et al. 2023](https://lmsys.org/blog/2023-03-30-vicuna/)                                                                  |



&nbsp;
## General Instructions


### 1. List Available Models

To see all supported models, run the following command without arguments:

```bash
litgpt download
```

The output is shown below:

```
codellama/CodeLlama-13b-hf
codellama/CodeLlama-13b-Instruct-hf
codellama/CodeLlama-13b-Python-hf
codellama/CodeLlama-34b-hf
codellama/CodeLlama-34b-Instruct-hf
codellama/CodeLlama-34b-Python-hf
codellama/CodeLlama-70b-hf
codellama/CodeLlama-70b-Instruct-hf
codellama/CodeLlama-70b-Python-hf
codellama/CodeLlama-7b-hf
codellama/CodeLlama-7b-Instruct-hf
codellama/CodeLlama-7b-Python-hf
databricks/dolly-v2-12b
databricks/dolly-v2-3b
databricks/dolly-v2-7b
EleutherAI/pythia-1.4b
EleutherAI/pythia-1.4b-deduped
EleutherAI/pythia-12b
EleutherAI/pythia-12b-deduped
EleutherAI/pythia-14m
EleutherAI/pythia-160m
EleutherAI/pythia-160m-deduped
EleutherAI/pythia-1b
EleutherAI/pythia-1b-deduped
EleutherAI/pythia-2.8b
EleutherAI/pythia-2.8b-deduped
EleutherAI/pythia-31m
EleutherAI/pythia-410m
EleutherAI/pythia-410m-deduped
EleutherAI/pythia-6.9b
EleutherAI/pythia-6.9b-deduped
EleutherAI/pythia-70m
EleutherAI/pythia-70m-deduped
garage-bAInd/Camel-Platypus2-13B
garage-bAInd/Camel-Platypus2-70B
garage-bAInd/Platypus-30B
garage-bAInd/Platypus2-13B
garage-bAInd/Platypus2-70B
garage-bAInd/Platypus2-70B-instruct
garage-bAInd/Platypus2-7B
garage-bAInd/Stable-Platypus2-13B
google/gemma-2b
google/gemma-2b-it
google/gemma-7b
google/gemma-7b-it
lmsys/longchat-13b-16k
lmsys/longchat-7b-16k
lmsys/vicuna-13b-v1.3
lmsys/vicuna-13b-v1.5
lmsys/vicuna-13b-v1.5-16k
lmsys/vicuna-33b-v1.3
lmsys/vicuna-7b-v1.3
lmsys/vicuna-7b-v1.5
lmsys/vicuna-7b-v1.5-16k
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-13b-hf
meta-llama/Llama-2-70b-chat-hf
meta-llama/Llama-2-70b-hf
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-7b-hf
microsoft/phi-1_5
microsoft/phi-2
mistralai/Mistral-7B-Instruct-v0.1
mistralai/Mistral-7B-Instruct-v0.2
mistralai/Mistral-7B-v0.1
mistralai/Mixtral-8x7B-Instruct-v0.1
mistralai/Mixtral-8x7B-v0.1
NousResearch/Nous-Hermes-13b
NousResearch/Nous-Hermes-llama-2-7b
NousResearch/Nous-Hermes-Llama2-13b
openlm-research/open_llama_13b
openlm-research/open_llama_3b
openlm-research/open_llama_7b
stabilityai/FreeWilly2
stabilityai/stable-code-3b
stabilityai/stablecode-completion-alpha-3b
stabilityai/stablecode-completion-alpha-3b-4k
stabilityai/stablecode-instruct-alpha-3b
stabilityai/stablelm-3b-4e1t
stabilityai/stablelm-base-alpha-3b
stabilityai/stablelm-base-alpha-7b
stabilityai/stablelm-tuned-alpha-3b
stabilityai/stablelm-tuned-alpha-7b
stabilityai/stablelm-zephyr-3b
tiiuae/falcon-180B
tiiuae/falcon-180B-chat
tiiuae/falcon-40b
tiiuae/falcon-40b-instruct
tiiuae/falcon-7b
tiiuae/falcon-7b-instruct
TinyLlama/TinyLlama-1.1B-Chat-v1.0
TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
togethercomputer/LLaMA-2-7B-32K
togethercomputer/RedPajama-INCITE-7B-Base
togethercomputer/RedPajama-INCITE-7B-Chat
togethercomputer/RedPajama-INCITE-7B-Instruct
togethercomputer/RedPajama-INCITE-Base-3B-v1
togethercomputer/RedPajama-INCITE-Base-7B-v0.1
togethercomputer/RedPajama-INCITE-Chat-3B-v1
togethercomputer/RedPajama-INCITE-Chat-7B-v0.1
togethercomputer/RedPajama-INCITE-Instruct-3B-v1
togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1
Trelis/Llama-2-7b-chat-hf-function-calling-v2
```

&nbsp;
### 2. Download Model Weights

To download the weights for a specific model, use the `--repo_id` argument. Replace `<repo_id>` with the model's repository ID. For example:

```bash
litgpt download --repo_id <repo_id>
```
This command downloads the model checkpoint into the `checkpoints/` directory.

&nbsp;
### 3. Additional Help


For more options, add the `--help` flag when running the script:

```bash
litgpt download --help
```

&nbsp;
### 4. Run the Model

After conversion, run the model with the `--checkpoint_dir` flag, adjusting `repo_id` accordingly:

```bash
litgpt chat --checkpoint_dir checkpoints/<repo_id>
```

&nbsp;
## Tinyllama Example

This section shows a typical end-to-end example for downloading and using TinyLlama:

1. List available TinyLlama checkpoints:

```bash
litgpt download | grep Tiny
```

```
TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

2. Download a TinyLlama checkpoint:

```bash
export repo_id=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
litgpt download --repo_id $repo_id
```

3. Use the TinyLlama model:

```bash
litgpt chat --checkpoint_dir checkpoints/$repo_id
```

&nbsp;
## Specific Models

Note that certain models require that you've been granted access to the weights on the Hugging Face Hub. 

For example, to get access to the Gemma 2B model, you can do so by following the steps at https://huggingface.co/google/gemma-2b. After access is granted, you can find your HF hub token in https://huggingface.co/settings/tokens.

Once you've been granted access and obtained the access token you need to pass the additional `--access_token`:

```bash
litgpt download \
  --repo_id google/gemma-2b \
  --access_token your_hf_token
```

&nbsp;
## Finetunes and other model variants

Sometimes you want to download the weights of a finetune of one of the models listed above. To do this, you need to manually specifiy the `model_name` associated to the config to use. For example:

```bash
litgpt download \
  --repo_id NousResearch/Hermes-2-Pro-Mistral-7B \
  --model_name Mistral-7B-v0.1
```

&nbsp;
## Tips for GPU Memory Limitations

The `download.py` script will automatically convert the downloaded model checkpoint into a LitGPT-compatible format. In case this conversion fails due to GPU memory constraints, you can try to reduce the memory requirements by passing the  `--dtype bf16-true` flag to convert all parameters into this smaller precision (however, note that most model weights are already in a bfloat16 format, so it may not have any effect):


```bash
litgpt download \
  --repo_id <repo_id>
  --dtype bf16-true
```

(If your GPU does not support the bfloat16 format, you can also try a regular 16-bit float format via `--dtype 16-true`.)

&nbsp;
## Converting Checkpoints Manually

For development purposes, for example, when adding or experimenting with new model configurations, it may be beneficial to split the weight download and model conversion into two separate steps. 

You can do this by passing the `--convert_checkpoint false` option to the download script:

```bash
litgpt download \
  --repo_id <repo_id> \
  --convert_checkpoint false
```

and then calling the `convert_hf_checkpoint.py` script:

```bash
litgpt convert to_litgpt \
  --checkpoint_dir checkpoint_dir/<repo_id>
```

&nbsp;
## Downloading Tokenizers Only

In some cases we don't need the model weight, for example, when we are pretraining a model from scratch instead of finetuning it. For cases like this, you can use the `--tokenizer_only` flag to only download a model's tokenizer, which can then be used in the pretraining scripts:

```bash
litgpt download \
  --repo_id TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --tokenizer_only true
```

and

```bash
litgpt pretrain \
  --data ... \
  --model_name tiny-llama-1.1b \
  --tokenizer_dir checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/
```
