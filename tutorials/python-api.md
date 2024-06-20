# LitGPT Python API

This is a work-in-progress draft describing the current LitGPT Python API (experimental and subject to change).


&nbsp;
## Model loading

Download a model using the CLI:

```bash
litgpt download EleutherAI/pythia160-m
```

Then, load the model in Python:

```python
from litgpt.api import LLM
llm = LLM.load("EleutherAI/pythia-160m", accelerator="cuda", devices=1)
```

&nbsp;
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

&nbsp;
## Pretraining

To start with random weights, initialize the model with `from_checkpoint=False`. Note that this requires passing a `tokenizer_dir` that contains a valid tokenizer file. 

```python
from litgpt.api import LLM
llm = LLM.load("pythia-160m", accelerator="cuda", devices=1, from_checkpoint=False, tokenizer_dir="EleutherAI/pythia-160m")
```

To be continued ...