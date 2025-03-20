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
    "microsoft/phi-2",
    distribute=None
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
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        distribute=None
    )

    llm.distribute(generate_strategy="tensor_parallel", devices=4)

    print(llm.generate(prompt="What do llamas eat?"))
    print(llm.generate(prompt="What is 1+2?", top_k=1))
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

# Using 1 device(s)
#  Llamas are herbivores and primarily eat grass, leaves, and shrubs. They have a unique digestive system that allows them to efficiently extract nutrients from tough plant material.

# {'Inference speed in tokens/sec': [17.617540650112936],
#  'Seconds to first token': [0.6533610639999097],
#  'Seconds total': [1.4758019020000575],
#  'Tokens generated': [26],
#  'Total GPU memory allocated in GB': [5.923729408]}
```

To get more reliably estimates, it's recommended to repeat the benchmark for multiple iterations via `num_iterations=10`:

```python
text, bench_d = llm.benchmark(num_iterations=10, prompt="What do llamas eat?", top_k=1, stream=True)
print(text)
pprint(bench_d)

# Using 1 device(s)
#  Llamas are herbivores and primarily eat grass, leaves, and shrubs. They have a unique digestive system that allows them to efficiently extract nutrients from tough plant material.

# {'Inference speed in tokens/sec': [17.08638672485105,
#                                    31.79908547222976,
#                                    32.83646959864293,
#                                    32.95994240022436,
#                                    33.01563039816964,
#                                    32.85263413816648,
#                                    32.82712094713627,
#                                    32.69216141907453,
#                                    31.52431714347663,
#                                    32.56752130561681],
#  'Seconds to first token': [0.7278506560005553,
#                             0.022963577999689733,
#                             0.02399449199947412,
#                             0.022921959999621322,
# ...
```

As one can see, the first iteration may take longer due to warmup times. So, it's recommended to discard the first iteration:

```python
for key in bench_d:
    bench_d[key] = bench_d[key][1:]
```

For better visualization, you can use the `benchmark_dict_to_markdown_table` function

```python
from litgpt.api import benchmark_dict_to_markdown_table

print(benchmark_dict_to_markdown_table(bench_d_list))
```

| Metric                              | Mean                        | Std Dev                     |
|-------------------------------------|-----------------------------|-----------------------------|
| Seconds total                       | 0.80                        | 0.01                        |
| Seconds to first token              | 0.02                        | 0.00                        |
| Tokens generated                    | 26.00                       | 0.00                        |
| Inference speed in tokens/sec       | 32.56                       | 0.50                        |
| Total GPU memory allocated in GB    | 5.92                        | 0.00                        |


&nbsp;
# PyTorch Lightning Trainer support

You can use the LitGPT `LLM` class with the [PyTorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) to pretrain and finetune models.

The examples below show the usage via a simple 160 million parameter model for demonstration purposes to be able to quickly try it out. However, you can replace the `EleutherAI/pythia-160m` model with any model supported by LitGPT (you can find a list of supported models by executing `litgpt download list` or visiting the [model weight docs](download_model_weights.md)).

&nbsp;
## Step 1: Define a `LightningModule`

First, we define a `LightningModule` similar to what we would do when working with other types of neural networks in PyTorch Lightning:


```python
import torch
import litgpt
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L


class LitLLM(L.LightningModule):
    def __init__(self, checkpoint_dir, tokenizer_dir=None, trainer_ckpt_path=None):
        super().__init__()

        self.llm = LLM.load(checkpoint_dir, tokenizer_dir=tokenizer_dir, distribute=None)
        self.trainer_ckpt_path = trainer_ckpt_path

    def setup(self, stage):
        self.llm.trainer_setup(trainer_ckpt=self.trainer_ckpt_path)

    def training_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("validation_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]
```

In the code example above, note how we set `distribute=None` in `llm.load()` in the `__init__` method. This step is necessary because we want to let the PyTorch Lightning Trainer handle the GPU devices. We then call `self.llm.trainer_setup` in the `setup()` method, which adjusts the LitGPT settings to be compatible with the Trainer. Other than that, everything else looks like a standard `LightningModule`.

Next, we have a selection of different use cases, but first, let's set some general settings to specify the batch size and gradient accumulation steps:

```python
batch_size = 8
accumulate_grad_batches = 1
```

For larger models, you may want to decrease the batch size and increase the number of accumulation steps. (Setting `accumulate_grad_batches = 1` effectively disables gradient accumulation, and it is only shown here for reference in case you wish to change this setting.)

## Step 2: Using the Trainer

&nbsp;
### Use case 1: Pretraining from random weights

In case you plan to train a model from scratch (not recommended over finetuning because training a model from scratch in general requires substantial time and resources), you can do it as follows:

```python
# Create model with random as opposed to pretrained weights
llm = LLM.load("EleutherAI/pythia-160m", tokenizer_dir="EleutherAI/pythia-160m", init="random")
llm.save("pythia-160m-random-weights")
del llm

lit_model = LitLLM(checkpoint_dir="pythia-160m-random-weights", tokenizer_dir="EleutherAI/pythia-160m")
data = Alpaca2k()

data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)

trainer = L.Trainer(
    devices=1,
    accelerator="cuda",
    max_epochs=1,
    accumulate_grad_batches=accumulate_grad_batches,
    precision="bf16-true",
)
trainer.fit(lit_model, data)

lit_model.llm.model.to(lit_model.llm.preprocessor.device)
lit_model.llm.generate("hello world")
```

&nbsp;
### Use case 2: Continued pretraining or finetuning a downloaded model

The continued pretraining or finetuning from a downloaded model checkpoint is similar to the example above, except that we can skip the initial steps of instantiating a model with random weights.

```python

lit_model = LitLLM(checkpoint_dir="EleutherAI/pythia-160m")
data = Alpaca2k()

data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)

trainer = L.Trainer(
    devices=1,
    accelerator="cuda",
    max_epochs=1,
    accumulate_grad_batches=accumulate_grad_batches,
    precision="bf16-true",
)
trainer.fit(lit_model, data)

lit_model.llm.model.to(lit_model.llm.preprocessor.device)
lit_model.llm.generate("hello world")
```

&nbsp;
### Use case 3: Resume training from Trainer checkpoint

Suppose you trained a model and decide to follow up with a few additional training rounds. This can be achieved as follows by loading an existing Trainer checkpoint:

```python

import os

def find_latest_checkpoint(directory):
    latest_checkpoint = None
    latest_time = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ckpt'):
                file_path = os.path.join(root, file)
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_checkpoint = file_path

    return latest_checkpoint

lit_model = LitLLM(checkpoint_dir="EleutherAI/pythia-160m", trainer_ckpt_path=find_latest_checkpoint("lightning_logs"))

data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)

trainer = L.Trainer(
    devices=1,
    accelerator="cuda",
    max_epochs=1,
    accumulate_grad_batches=accumulate_grad_batches,
    precision="bf16-true",
)
trainer.fit(lit_model, data)

lit_model.llm.model.to(lit_model.llm.preprocessor.device)
lit_model.llm.generate("hello world")
```

&nbsp;
### Use case 4: Resume training after saving a checkpoint manually

This example illustrates how we can save a LitGPT checkpoint from a previous training run that we can load and use later. Note that compared to using the Trainer checkpoint in the previous section, the model saved via this approach also contains the tokenizer and other relevant files. Hence, this approach does not require the original `"EleutherAI/pythia-160m"` model checkpoint directory.

```python
lit_model.llm.save("finetuned_checkpoint")
del lit_model
lit_model = LitLLM(checkpoint_dir="finetuned_checkpoint")

data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)

trainer = L.Trainer(
    devices=1,
    accelerator="cuda",
    max_epochs=1,
    accumulate_grad_batches=accumulate_grad_batches,
    precision="bf16-true",
)
trainer.fit(lit_model, data)

lit_model.llm.model.to(lit_model.llm.preprocessor.device)
lit_model.llm.generate("hello world")
```
