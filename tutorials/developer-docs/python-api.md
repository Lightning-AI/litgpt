# LitGPT High-level Python API

This is a work-in-progress draft for a high-level LitGPT Python API.

&nbsp;
## Model loading & saving

The `LLM.load` command loads an `llm` object, which contains both the model object (a PyTorch module) and a preprocessor.

```python
from litgpt import LLM

llm = LLM.load(
    model="url | local_path",
    # high-level user only needs to care about those:
    memory_reduction="none | medium | strong"
    # advanced options for technical users:
    source="hf | local | other"
    quantize="bnb.nf4",
    precision="bf16-true",
    device=""auto | cuda | cpu",
)
```

Here,

-  `llm.model` contains the PyTorch Module
- and `llm.preprocessor.tokenizer`  contains the tokenizer

The `llm.save` command saves the model weights, tokenizer, and configuration information.


```python
llm.save(checkpoint_dir, format="lightning | ollama | hf")
```


&nbsp;
## Inference / Chat

```
response = llm.generate(
    prompt="What do Llamas eat?",
    temperature=0.1,
    top_p=0.8,
    ...
)
```


&nbsp;
## Dataset

The `llm.prepare_dataset` command prepares a dataset for training.

```
llm.download_dataset(
    URL,
    ...
)
```

```
dataset = llm.prepare_dataset(
    path,
    task="pretrain | instruction_finetune",
    test_portion=0.1,
    ...
)
```

&nbsp;
## Training


```python
llm.instruction_finetune(
    config=None,
    dataset=dataset,
    max_iter=10,
    method="full | lora | adapter | adapter_v2"
)
```

```python
llm.pretrain(config=None, dataset=dataset, max_iter=10, ...)
```

&nbsp;
## Serving


```python
llm.serve(port=8000)
```

Then in another Python session:

```python
import requests, json

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"prompt": "Fix typos in the following sentence: Example input"}
)

print(response.json()["output"])
```
