# Adding New Models

This document provides an overview and explanation of how new LLM architectures and model weights can be added to LitGPT.

&nbsp;

> [!NOTE]
> One of the design focus areas of LitGPT is to provide efficient readable code. At the same time, LitGPT aims to support selected LLMs that are useful to the community. LitGPT aims to reuse and share as much code as possible between different LLMs to strike a balance between code readability and enabling support for various LLMs. In short, we try to minimize writing custom code for a given LLM and aim for code reuse.


&nbsp;

&nbsp;
## 1. Discuss the LLM to be added

As an open-source project, we appreciate your contributions! However, before you begin putting valuable time and work into a contribution, ideally, open an issue to discuss whether support for a certain model is within the project's scope.

&nbsp;
## 2. Set up your development environment

Clone the repository:

```bash
git clone https://github.com/Lightning-AI/litgpt.git
```

Then, install it with the "editable" mode for development:

```bash
cd litgpt
pip install litgpt -e ".[all]"
```

&nbsp;
## 3. Update the config file

Update the [litgpt/config.py](../../litgpt/config.py) config file, adding the new model configuration there. It's easiest to start with the most similar model, copy the configuration, and then modify it according to the `config.json` file on the HF hub.

For example, suppose an entry for Llama 3 8B already exists and you want to add support for Llama 3 70B.

Copy the Llama 3 8B entry:

```python
 # https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
 dict(
     name="Llama-3-8B{}",
     hf_config=dict(org="meta-llama", name="Meta-Llama-3-8B{}"),
     vocab_size=128256,
     padding_multiple=64,
     n_layer=32,
     n_head=32,
     n_query_groups=8,
     rotary_percentage=1.0,
     parallel_residual=False,
     bias=False,
     norm_class_name="RMSNorm",
     mlp_class_name="LLaMAMLP",
     intermediate_size=14336,
     rope_base=500000,
 ),
```

Then create the entry for the 70B model. Here, make sure you update the values according to the `config.json` file available on the HF hub:

```python
# https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json
 dict(
     name="Llama-3-70B{}",
     hf_config=dict(org="meta-llama", name="Meta-Llama-3-70B{}"),
     vocab_size=128256,
     padding_multiple=64,
     n_layer=80,
     n_head=64,
     n_embd=8192,
     n_query_groups=8,
     rotary_percentage=1.0,
     parallel_residual=False,
     bias=False,
     norm_class_name="RMSNorm",
     mlp_class_name="LLaMAMLP",
     intermediate_size=28672,
     rope_base=500000,
 ),
```

&nbsp;

> [!NOTE]
> Some models may require you to implement a new MLP class analogous to `class LLaMAMLP`.
> A more or less reliable indicator is the presence of a `modeling.py` file in the model's original repository.
> If this file exists, it suggests that this model requires custom code.
> This will then also require additional changes beyond simply updating
> the configuration in LitGPT's `config.py`.

&nbsp;
## 4. Try downloading the model

After making the modifications above, try downloading the model:

```bash
litgpt download meta-llama/Meta-Llama-3-70B --access_token ...
```

&nbsp;

> [!NOTE]
> Not all models require an access token

&nbsp;

If the conversion following the download fails, proceed with the next section.

&nbsp;
## 5. Update the checkpoint conversion script

If the `litgpt download ...` command from the previous section failed, you may have to adjust the checkpoint conversion script: [litgpt/scripts/convert_hf_checkpoint.py](../../litgpt/scripts/convert_hf_checkpoint.py).

Here, you may have to adjust or implement a new `def copy_weights_hf_...` function.

You can test the updated conversion code without needing to redownload the weights as follows:

```bash
python litgpt/scripts/convert_hf_checkpoint.py meta-llama/Meta-Llama-3-70B
```

&nbsp;
## 6. Add the Prompt Style

If you are adding a new model class, find out its prompt style. First, check [litgpt/prompts.py](../../litgpt/prompts.py) if a similar prompt style template already exists. For Llama 3, this is as follows:

```python
class Llama3(PromptStyle):
     def apply(self, prompt: str, **kwargs: str) -> str:
         # https://github.com/meta-llama/llama3/blob/359887376f0aaf30e433f23e25df858d8c2a9833/llama/tokenizer.py#L202-L229
         return (
             "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
             "You are a helpful assistant.<|eot_id|>\n"  # The system prompt is optional
             "<|start_header_id|>user<|end_header_id|>\n\n"
             f"{prompt}<|eot_id|>\n"
             "<|start_header_id|>assistant<|end_header_id|>\n\n"
         )

     def stop_tokens(self, tokenizer: "Tokenizer") -> Tuple[List[int], ...]:
         return (
             [tokenizer.eos_id],
             [tokenizer.token_to_id("<|eot_id|>")],
         )
```

If your model requires a different prompt template, create a new `PromptStyle` class.

Then, in the same file, update the `prompt_styles` dictionary:

```python
prompt_styles: Dict[str, Type[PromptStyle]] = {
    ...
    "llama3": Llama3,
}
```

Finally, also in the same file, update the `model_name_to_prompt_style` function:

```python
def model_name_to_prompt_style(model_name: str) -> PromptStyle:
    ...
    if re.search("Llama-3.*-Instruct", model_name):
    return Llama3()
```

&nbsp;
## 7. Try using the model for inference

Next, use the model to see if inference works:

```bash
litgpt generate meta-llama/Meta-Llama-3-70B
```

&nbsp;

> [!NOTE]
> If you notice that the model produces non-sensible language outputs, you need to double-check the config file and find out if there are incorrect values or other problems. The next section on adding unit tests may offer additional pointers.

&nbsp;

&nbsp;
## 8. Add unit tests

&nbsp;
### 8.1 Add model unit tests

Open the [`tests/test_model.py`](../../tests/test_model.py) file and add a new `def test_against_hf_...` function using one of the existing functions as a template. For instance,

```python
def test_against_hf_llama2(ours_kwargs, device, dtype):
...
    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)
```

If the

```bash
litgpt generate meta-llama/Meta-Llama-3-70B
```

command from the previous section produces incoherent text, this function can be a helpful guide for debugging. For this, modify the implementation in `transformers` and `litgpt` packages (on your local installation), to inspect or print out the intermediate values at a layer. It's recommend starting with the embedding layers and then go through one layer at the time, to find out where the values differ to get pointers for debugging.

Test the unit test via

```python
pytest tests/test_model.py::test_against_hf_...
```

&nbsp;
### 8.2 Add prompt style unit test

Open the [`tests/test_model.py`](../../tests/test_model.py) file and add a test for the respective prompts you added earlier, if applicable. For example,


```python
def test_prompt_style_from_config():
    model_names = [
        ...
        "Llama-3-70B-Instruct",
        ...
    ]
```

Run the unit test via

```python
pytest tests/test_prompts.py
```

&nbsp;
## 9. Try finetuning the model

Now, try finetuning the model:

```bash
litgpt finetune meta-llama/Meta-Llama-3-70B --train.max_steps 10
```

&nbsp;
## 10. Update the documentation

Finally, update the documentation files.

&nbsp;
### 10.1 Update the README file

Update the "All Models" table in the [README.md](../../README.md) file.

&nbsp;
### 10.2 Update the download tutorials

Add the new model to the model table at the top as well as to the list under `litgpt download list`.
