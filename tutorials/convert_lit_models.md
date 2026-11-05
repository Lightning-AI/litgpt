## Converting LitGPT weights to Hugging Face Transformers

LitGPT weights need to be converted to a format that Hugging Face understands with a [conversion script](../litgpt/scripts/convert_lit_checkpoint.py) before our scripts can run.

We provide a helpful command to convert models LitGPT models back to their equivalent Hugging Face Transformers format:

```bash
litgpt convert_from_litgpt checkpoint_dir converted_dir
```

These paths are just placeholders, you will need to customize them based on which finetuning or pretraining command you ran and its configuration.

### Loading converted LitGPT checkpoints into transformers

The conversion saves the weights as a `pytorch_model.bin` file and copies the configuration and tokenizer files
(such as `config.json` and `tokenizer.json`) from the source checkpoint directory. You can then load the converted
checkpoint in a Python session as follows:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("converted_dir/", local_files_only=True)
```



### Merging LoRA weights

Please note that if you want to convert a model that has been finetuned using an adapter like LoRA, these weights should be [merged](../litgpt/scripts/merge_lora.py) to the checkpoint prior to converting.

```sh
litgpt merge_lora path/to/lora/checkpoint_dir
```

<br>
<br>

# A finetuning and conversion tutorial

This section contains a reproducible example for finetuning a LitGPT model and converting it back into a HF `transformer` model.

1. Download a model of interest:

For convenience, we first specify an environment variable (optional) to avoid copy and pasting the whole path:

```bash
export repo_id=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

Instead of using TinyLlama, you can replace the `repo_id` target with any other model repository
specifier that is currently supported by LitGPT. You can get a list of supported repository specifier
by running `litgpt/scripts/download.py` without any additional arguments.

Then, we download the model we specified via `$repo_id` above:

```bash
litgpt download $repo_id
```

2. Finetune the model:


```bash
export finetuned_dir=out/lit-finetuned-model

litgpt finetune_lora $repo_id \
   --out_dir $finetuned_dir \
   --train.epochs 1 \
   --data Alpaca
```

3. Merge LoRA weights:

Note that this step only applies if the model was finetuned with `lora.py` above and not when `full.py` was used for finetuning.

```bash
litgpt merge_lora $finetuned_dir/final
```


4. Convert the finetuning model back into a HF format:

```bash
litgpt convert_from_litgpt $finetuned_dir/final/ out/hf-tinyllama/converted
```


5. Load the model into a `transformers` model:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("out/hf-tinyllama/converted/", local_files_only=True)
```

&nbsp;
## Using the LM Evaluation Harness

To evaluate LitGPT models, use the integrated evaluation utilities based on Eleuther AI's LM Evaluation Harness. For more information, please see the [evaluation](evaluation.md) documentation.

Alternatively, if you wish to use converted LitGPT models with the LM Evaluation Harness from [Eleuther AI's GitHub repository](https://github.com/EleutherAI/lm-evaluation-harness), you can use the following steps.

1. Follow the instructions above to load the model into a Hugging Face transformers model.

2. Create a `model.safetensor` file:

```python
model.save_pretrained("out/hf-tinyllama/converted/")
```

3. Run the evaluation harness, for example:

```bash
lm_eval --model hf \
    --model_args pretrained=out/hf-tinyllama/converted \
    --tasks "hellaswag,gsm8k,truthfulqa_mc2,mmlu,winogrande,arc_challenge" \
    --device "cuda:0" \
    --batch_size 4
```
