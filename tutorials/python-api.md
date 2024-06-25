# LitGPT Python API

This is a work-in-progress draft describing the current LitGPT Python API (experimental and subject to change).


## Model loading


```python
from litgpt import LLM
llm = LLM.load("microsoft/phi-2", accelerator="cuda")
```

&nbsp;
> [!TIP]
> The command above will download the model automatically if it doesn't already exist at `"microsoft/phi-2"` or `"checkpoints/microsoft/phi-2"`. If you want to suppress automatic downloads, pass the additional `init="local"` setting: `llm = LLM.load("microsoft/phi-2", accelerator="cuda", init="local")`
&nbsp;

&nbsp;
> [!NOTE]
> To get a list of all supported models, execute `litgpt download list` in the command line terminal.
&nbsp;

## Generate/Chat

Generate output using the `.generate` method:

```python
text = llm.generate("What do Llamas eat?", top_k=1, max_new_tokens=30)
print(text)
```

```
Llamas are herbivores and primarily eat grass, leaves, and shrubs. They have a specialized digestive system that allows them to efficiently extract
```

Alternative, stream the response one token at a time:

```python
result = llm.generate("hi", stream=True)
for e in result:
    print(e, end="", flush=True)
```

```
Llamas are herbivores and primarily eat grass, leaves, and shrubs. They have a specialized digestive system that allows them to efficiently extract
```


&nbsp;
## Random weights

To start with random weights, for example, if you plan a pretraining script, initialize the model with `init="random""`. Note that this requires passing a `tokenizer_dir` that contains a valid tokenizer file. 

```python
from litgpt.api import LLM
llm = LLM.load("pythia-160m", accelerator="cuda", init="random", tokenizer_dir="EleutherAI/pythia-160m")
```
