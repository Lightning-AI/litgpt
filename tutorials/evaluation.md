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
litgpt download microsoft/phi-2
```

The download command above will save the model to the `checkpoints/microsoft/phi-2` directory, which we can
specify in the following evaluation command:


```
litgpt evaluate microsoft/phi-2/ \
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
conversion steps if you run the `litgpt evaluate` on the same checkpoint directory again.

In some cases, for example, if you modified the model in the `checkpoint_dir` since the first `litgpt evaluate`
call, you need to use the `--force_conversion` flag to to update the files used by litgpt evaluate accordingly:

```
litgpt evaluate microsoft/phi-2/ \
  --batch_size 4 \
  --out_dir evaluate_model/ \
  --tasks "hellaswag,truthfulqa_mc2,mmlu" \
  --force_conversion true
```

&nbsp;

> [!TIP]
> Run `litgpt evaluate list` to print a list
> of the supported tasks. To filter for a specific subset of tasks, e.g., MMLU, use `litgpt evaluate list | grep mmlu`.

> [!TIP]
> The evaluation may take a long time, and for testing purpoes, you may want to reduce the number of tasks
> or set a limit for the number of examples per task, for example, `--limit 10`.




&nbsp;

### Evaluating LoRA-finetuned LLMs

No further conversion is necessary when evaluating LoRA-finetuned models as the `finetune_lora` command already prepares the necessary merged model files:

```bash
litgpt finetune_lora microsoft/phi-2 \
  --out_dir lora_model
```

&nbsp;

```bash
litgpt evaluate lora_model/final \
  --batch_size 4 \
  --tasks "hellaswag,truthfulqa_mc2,mmlu" \
  --out_dir evaluate_model/ \
```


&nbsp;

### Evaluating on a custom test set

There is currently no built-in function to evaluate models on custom test sets. However, this section describes a general approach that users can take to evaluate the responses of a model using another LLM.

Suppose you have a test dataset with the following structure:

```python
test_data = [
    {
        "instruction": "Name the author of 'Pride and Prejudice'.",
        "input": "",
        "output": "Jane Austen."
    },
    {
        "instruction": "Pick out the adjective from the following list.",
        "input": "run, tall, quickly",
        "output": "The correct adjective from the list is 'tall.'"
    },
]
```

For simplicity, the dictionary above only contains two entries. In practice, it is recommended to use test datasets that contain at least 100 entries (ideally 1000 or more).

If your dataset is stored in JSON format, use the following code to load it:

```python
with open("test_data.json", "r") as file:
    test_data = json.load(file)
```

Next, it is recommended to format the dataset according to a prompt style. For example, to use the `Alpaca` prompt style, use the following code:

```python
from litgpt.prompts import Alpaca

prompt_style = Alpaca()
prompt_style.apply(prompt=test_data[0]["instruction"], **test_data[0])
```

which returns

```
"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nName the author of 'Pride and Prejudice'.\n\n### Response:\n
```

Next, load the LLM you want to evaluate. For this example, we use `phi-2`:

```python
from litgpt import LLM

llm = LLM.load("microsoft/phi-2")
```

Then, using the loaded model, we add the test set responses to the dataset:


```python
from tqdm import trange


for i in trange(len(test_data)):
    response = llm.generate(prompt_style.apply(prompt=test_data[i]["instruction"], **test_data[i]))
    test_data[i]["response"] = response
```

Next, we use a second LLM to calculate the response quality on a scale from 0 to 100. It is recommended to use the 70B Llama 3 instruction-fintuned model for this task, or the smaller 8B Llama 3 model, which is more resource-efficient:


```python
del llm # delete previous `llm` to free up GPU memory
scorer = LLM.load("meta-llama/Meta-Llama-3-8B-Instruct", access_token="...")
```

Then, based on this LLM, we calculate the response quality with the following function:

```python
from tqdm import tqdm


def generate_model_scores(data_dict, model, response_field="response", target_field="output"):
    scores = []
    for entry in tqdm(data_dict, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry[target_field]}`, "
            f"score the model response `{entry[response_field]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = model.generate(prompt, max_new_tokens=50)
        try:
            scores.append(int(score))
        except ValueError:
            continue

    return scores
```


```python
scores = generate_model_scores(test_data, model=scorer)
print(f"\n{llm}")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")
```

This will print out the average score on all test set entries:

```
Scoring entries: 100%|██████████| 2/2 [00:00<00:00,  4.37it/s]

Number of scores: 2 of 2
Average score: 47.50
```
