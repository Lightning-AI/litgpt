# LitGPT Python API

This is a work-in-progress draft describing the current LitGPT Python API (experimental and subject to change).


## Model loading

Download a model using the CLI:

```bash
litgpt download microsoft/phi-2
```

Then, load the model in Python:

```python
from litgpt import LLM
llm = LLM.load("microsoft/phi-2", accelerator="cuda")
```

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