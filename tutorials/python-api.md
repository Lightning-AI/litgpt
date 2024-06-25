# LitGPT Python API

This is a work-in-progress draft describing the current LitGPT Python API (experimental and subject to change).


## Model loading

Use the `LLM.load` method to load a model from a LitGPT model checkpoint folder. For example, consider loading a Phi-2 model. If a given checkpoint directory `"microsoft/phi-2"` does not exist as a local checkpoint directory, the model will be downloaded automatically from the HF Hub (assuming that `"microsoft/phi-2"` is a valid repository name):

```python
from litgpt import LLM

llm_1 = LLM.load("microsoft/phi-2")
```

```
config.json: 100%|████████████████████████████████████████████████| 735/735 [00:00<00:00, 7.75MB/s]
generation_config.json: 100%|█████████████████████████████████████| 124/124 [00:00<00:00, 2.06MB/s]
model-00001-of-00002.safetensors: 100%|███████████████████████████| 5.00G/5.00G [00:12<00:00, 397MB/s]
model-00002-of-00002.safetensors: 100%|███████████████████████████| 564M/564M [00:01<00:00, 421MB/s]
model.safetensors.index.json: 100%|███████████████████████████████| 35.7k/35.7k [00:00<00:00, 115MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████| 2.11M/2.11M [00:00<00:00, 21.5MB/s]
tokenizer_config.json: 100%|██████████████████████████████████████| 7.34k/7.34k [00:00<00:00, 80.6MB/s]
```

&nbsp;
> [!NOTE]
> To get a list of all supported models, execute `litgpt download list` in the command line terminal.
&nbsp;
<br>


If you attempt to load the model again, LitGPT will load this model from a local directory since it's already been downloaded:

```python
llm_2 = LLM.load("microsoft/phi-2")
```


If you created a pretrained of finetuned model checkpoint via LitGPT, you can load it in a similar fashion:

```python
my_llm = LLM.load("path/to/my/local/checkpoint")
```




&nbsp;
## Generate/Chat

Generate output using the `.generate` method:

```python
from litgpt import LLM

llm = LLM.load("microsoft/phi-2")

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
llm = LLM.load("pythia-160m", init="random", tokenizer_dir="EleutherAI/pythia-160m")
```
