# LLM Evaluation

&nbsp;

## Using lm-evaluation-harness

You can evaluate LitGPT using [EleutherAI's lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) framework with a large number of different evaluation tasks.

You need to install the `lm-eval` framework first:

```bash
pip install lm_eval
```

&nbsp;

### Evaluating LitGPT base models

Suppose you downloaded a base model that we want to evaluate. Here, we use the `microsoft/phi-2` model:

```bash
litgpt download --repo_id microsoft/phi-2
```

The download command above will save the model to the `checkoints/microsoft/phi-2` directory, which we can
specify in the following evaluation command:


```
litgpt evaluate \
  --checkpoint_dir checkpoints/microsoft/phi-2/ \
  --batch_size 4 \
  --tasks "hellaswag,truthfulqa_mc2,mmlu" \
  --out_dir evaluate_model/
```

The resulting output is as follows:

```
...
|---------------------------------------|-------|------|-----:|--------|-----:|---|-----:|
...
|truthfulqa_mc2                         |      2|none  |     0|acc     |0.4656|±  |0.0164|
|hellaswag                              |      1|none  |     0|acc     |0.2569|±  |0.0044|
|                                       |       |none  |     0|acc_norm|0.2632|±  |0.0044|

|      Groups      |Version|Filter|n-shot|Metric|Value |   |Stderr|
|------------------|-------|------|-----:|------|-----:|---|-----:|
|mmlu              |N/A    |none  |     0|acc   |0.2434|±  |0.0036|
| - humanities     |N/A    |none  |     0|acc   |0.2578|±  |0.0064|
| - other          |N/A    |none  |     0|acc   |0.2401|±  |0.0077|
| - social_sciences|N/A    |none  |     0|acc   |0.2301|±  |0.0076|
| - stem           |N/A    |none  |     0|acc   |0.2382|±  |0.0076|
```


Please note that the `litgpt evaluate` command run an internal model conversion. 
This is only necessary the first time you want to evaluate a model, and it will skip the
conversion steps if you run the `litgpt evaluate` on the same checkpint directory again.

In some cases, for example, if you modified the model in the `checkpoint_dir` since the first `litgpt evaluate`
call, you need to use the `--force_conversion` flag to to update the files used by litgpt evaluate accordingly: 

```
litgpt evaluate \
  --checkpoint_dir checkpoints/microsoft/phi-2/ \
  --batch_size 4 \
  --out_dir evaluate_model/ \
  --tasks "hellaswag,truthfulqa_mc2,mmlu" \
  --force_conversion true
```

&nbsp;

> [!TIP]
> Run `litgpt evaluate` without any additional arguments to print a list
> of the supported tasks. 

> [!TIP]
> The evaluation may take a long time, and for testing purpoes, you may want to reduce the number of tasks
> or set a limit for the number of examples per task, for example, `--limit 10`.




&nbsp;

### Evaluating LoRA-finetuned LLMs

No further conversion is necessary when evaluating LoRA-finetuned models as the `finetune lora` command already prepares the necessary merged model files:

```bash
litgpt finetune lora \
  --checkpoint_dir checkpoints/microsoft/phi-2 \
  --out_dir lora_model
```

&nbsp;

```bash
litgpt evaluate \
  --checkpoint_dir lora_model/final \
  --batch_size 4 \
  --tasks "hellaswag,truthfulqa_mc2,mmlu" \
  --out_dir evaluate_model/ \
```
