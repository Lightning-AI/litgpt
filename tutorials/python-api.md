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



&nbsp;
## Multi-GPU strategies

By default, the model is loaded onto a single GPU. Optionally, you can use the `.distribute()` method with the "sequential" or "tensor_parallel" `generate_strategy` settings.

### Sequential strategy

the `generate_strategy="sequential"` setting to load different parts of the models onto different GPUs. The goal behind this strategy is to support models that cannot fit into single-GPU memory. (Note that if you have a model that can fit onto a single GPU, this sequential strategy will be slower.)

```python
from litgpt.api import LLM

llm = LLM.load(
    "microsoft/phi-2"
)

llm.distribute(
    generate_strategy="sequential",
    devices=4,  # Optional setting, otherwise uses all available GPUs
    fixed_kv_cache_size=256  # Optionally use a small kv-cache to further reduce memory usage
)
```

```
Using 4 devices
Moving '_forward_module.transformer.h.31' to cuda:3: 100%|██████████| 32/32 [00:00<00:00, 32.71it/s]
```

After initializing the model, the model can be used via the `generate` method similar to the default `generate_strategy` setting:

```python
text = llm.generate("What do llamas eat?", max_new_tokens=100)
print(text)
```

```
 Llamas are herbivores and their diet consists mainly of grasses, plants, and leaves.
```

&nbsp;
### Tensor parallel strategy

The sequential strategy explained in the previous subsection distributes the model sequentially across GPUs, which allows users to load models that would not fit onto a single GPU. However, due to this method's sequential nature, processing is naturally slower than parallel processing. 

To take advantage of parallel processing via tensor parallelism, you can use the `generate_strategy="tensor_parallel" setting. However, this method has downsides: the initial setup may be slower for large models, and it cannot run in interactive processes such as Jupyter notebooks.

```python
from litgpt.api import LLM


if __name__ == "__main__":

    llm = LLM.load(
        model="microsoft/phi-2",
        distribute=None
    )

    llm.distribute(generate_strategy="tensor_parallel", devices=4)

    print(llm.generate(prompt="What do llamas eat?", top_k=1))
```


&nbsp;
## Speed and resource estimates

Use the `.benchmark()` method to compare the computational performance of different settings. The `.benchmark()` method takes the same arguments as the `.generate()` method. For example, we can estimate the speed and GPU memory consumption as follows (the resulting numbers were obtained on an A10G GPU):

```python
from litgpt.api import LLM
from pprint import pprint

llm = LLM.load(
    model="microsoft/phi-2",
    distribute=None
)

llm.distribute(fixed_kv_cache_size=500)

text, bench_d = llm.benchmark(prompt="What do llamas eat?", top_k=1, stream=True)
print(text)
pprint(bench_d)


# Llamas are herbivores and primarily eat grass, leaves, and shrubs. They have a specialized 
# digestive system that allows them to efficiently extract nutrients from plant material.

# {'Inference speed in tokens/sec': 15.687777681894985,
#  'Seconds to first token': 0.5756612900004257,
#  'Seconds total': 1.5935972900006163,
#  'Tokens generated': 25,
#  'Total GPU memory allocated in GB': 11.534106624}
```