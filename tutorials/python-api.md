# LitGPT Python API

This is a work-in-progress draft describing the current LitGPT Python API (experimental and subject to change).


## Model loading

Download a model using the CLI:

```bash
litgpt download EleutherAI/pythia160-m
```

Then, load the model in Python:

```python
from litgpt.api import LLM
llm = LLM.load("EleutherAI/pythia-160m", device_type="cuda", devices=1)
```

## Generate/Chat

Generate output using the `.generate` method:

```python
text = llm.generate("What do Llamas eat?", top_k=1)
print(text)
```

```
What do Llamas eat?

"A lot of people, the Llamas, I was, a Llamas, a lama, a lama, a lama, a lama, a lama, a lama, a lama, a
```