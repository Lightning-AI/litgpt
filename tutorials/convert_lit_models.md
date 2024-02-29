## Converting Lit-GPT weights to Hugging Face Transformers

Lit-GPT weights need to be converted to a format that Hugging Face understands with a [conversion script](../scripts/convert_lit_checkpoint.py) before our scripts can run.

We provide a helpful script to convert models Lit-GPT models back to their equivalent Hugging Face Transformers format:

```sh
python scripts/convert_lit_checkpoint.py \
    --checkpoint_path path/to/litgpt/model.pth \
    --output_path out/converted.ckpt \
    --config_path path/to/litgpt/config.json
```

These paths are just placeholders, you will need to customize them based on which finetuning or pretraining script you ran and it's configuration.

### Loading converted Lit-GPT checkpoints into transformers

If you want to load the converted checkpoints into a `transformers` model, please make sure you copied the original `config.json` file into the folder that contains the `converted.ckpt` file saved via `--output_path` above.

For example,

```bash
wget https://huggingface.co/repo_id/raw/main/config.json -O out/config.json
```

Then, you can load the checkpoint file in a Python session as follows:

```python
import torch
from transformers import AutoModel


state_dict = torch.load("out/converted.ckpt")
model = AutoModel.from_pretrained(
    "out/", local_files_only=True, state_dict=state_dict
)
```

Alternatively, you can also load the model without copying the `config.json` file as follows:

```python
model = AutoModel.from_pretrained("online_repo_id", state_dict=state_dict)
```



### Merging LoRA weights

Please note that if you want to convert a model that has been fine-tuned using an adapter like LoRA, these weights should be [merged](../scripts/merge_lora.py) to the checkpoint prior to converting.

```sh
python scripts/merge_lora.py \
    --checkpoint_dir path/to/litgpt_checkpoint/ \
    --lora_path path/to/litgpt/lora_finetuned.pth \
    --out_dir out/merged.ckpt
```

<br>
<br>

# A finetuning and conversion tutorial

This section contains a reproducible example for finetuning a Lit-GPT model and converting it back into a HF `transformer` model.

1. Download a model of interest:

For convenience, we first specify an environment variable (optional) to avoid copy and pasting the whole path:

```bash
export repo_id=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

Instead of using TinyLlama, you can replace the `repo_id` target with any other model repository 
specifier that is currently supported by Lit-GPT. You can get a list of supported repository specifier
by running `scripts/download.py` without any additional arguments.

Then, we download the model we specified via `$repo_id` above:

```bash
python scripts/download.py --repo_id $repo_id
```


2. Prepare a dataset for finetuning:

```bash
python scripts/prepare_alpaca.py \
    --checkpoint_dir checkpoints/$repo_id \
    --destination_path data/alpaca
```

3. Finetune the model:


```bash
export finetuned_dir=out/lit-finetuned-model

python finetune/lora.py \
   --io.checkpoint_dir checkpoints/$repo_id \
   --io.train_data_dir data/alpaca \
   --io.val_data_dir data/alpaca \
   --train.epochs 1 \
   --io.out_dir $finetuned_dir
```

4. Merge LoRA weights:

Note that this step only applies if the model was finetuned with `lora.py` above and not when `full.py` was used for finetuning.

```bash
python scripts/merge_lora.py \
    --checkpoint_dir checkpoints/$repo_id \
    --lora_path $finetuned_dir/lit_model_lora_finetuned.pth \
    --out_dir $finetuned_dir/merged/
```


5. Convert the finetuning model back into a HF format:

```bash
python scripts/convert_lit_checkpoint.py \
   --checkpoint_path $finetuned_dir/merged/lit_model.pth \
   --output_path out/hf-tinyllama/converted_model.pth \
   --config_path checkpoints/$repo_id/lit_config.json 
```

(If you used `full.py` instead of `lora.py` to finetune your model, 
replace `$finetuned_dir/merged/lit_model.pth` with `$finetuned_dir/lit_model_finetuned.pth`.)


6. Load the model into a `transformers` model:

```python
import torch
from transformers import AutoModel

state_dict = torch.load('out/hf-tinyllama/converted_model.pth')
model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", state_dict=state_dict)
```
