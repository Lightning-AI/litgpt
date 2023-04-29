# Full Finetuning

Full finetuning updates all layers in the pretrained LLaMA model. This *regular* finetuning procedure is typically considered as the baseline for parameter-efficient alternatives such as Low-Rank Adaptation (LoRA) or LLaMA-Adapter.

The current  [finetune_full.py](../scripts/finetune_full.py) we provide uses 4 A100 GPUs with a fully-sharded data parallel strategy to finetune Lit-LLaMA 7B on [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset. The A100 GPUs have 40 GB each, but it may require less memory to finetune this model.



## Preparation

The steps here only need to be done once:

1. Follow the instructions in the [README](README.md) to install the dependencies.

2. Download and convert the weights and save them in the `./checkpoints` folder as described [here](download_weights.md).

4. Download the data and generate the Alpaca instruction tuning dataset:

   ```bash
   python scripts/prepare_alpaca.py
   ```

   or [prepare your own dataset](#tune-on-your-own-dataset).

## Running the finetuning

```bash
python finetune_full.py
```


You can speed up training by setting the `devices` variable in the script to utilize more GPUs if available or increase the `batch_size`.
Depending on the available GPU memory, you can also tune the `micro_batch_size` parameter to utilize the GPU efficiently.

For example, the following settings will let you finetune the model in 32 hours using a fully-sharded data parallel strategy:
```python
devices = 4
batch_size = 128 // devices
micro_batch_size = 4
```

This script will save checkpoints periodically to the folder `out/`.

> **Note**
> All scripts support argument [customization](customize_paths.md)

## Test the model

You can test the finetuned model with your own instructions by running:

```bash
python generate_full.py \
    --prompt "Recommend a movie to watch on the weekend." \
    --quantize llm.int8
```
Output:
```
A good movie to watch on the weekend would be The Lion King, since it's a classic family film that everyone can enjoy...
```
If your GPU supports `bfloat16`, the script will automatically use it. Together with `--quantize llm.int8`, this brings the memory consumption down to ~8 GB.

## Tune on your dataset

With only a few modifications, you can prepare and train on your own instruction dataset.

1. Create a json file in which each row holds one instruction-response pair. 
   A row has an entry for 'instruction', 'input', and 'output', where 'input' is optional an can be 
   the empty string if the instruction doesn't require a context. Below is an example json file:

    ```
    [
        {
            "instruction": "Arrange the given numbers in ascending order.",
            "input": "2, 4, 0, 8, 3",
            "output": "0, 2, 3, 4, 8"
        },
        ...
    ]
    ```

2. Make a copy of `scripts/prepare_alpaca.py` and name it what you want:

    ```bash
    cp scripts/prepare_alpaca.py scripts/prepare_mydata.py
    ```

3. Modify `scripts/prepare_mydata.py` to read the json data file.
4. Run the script to generate the preprocessed, tokenized train-val split:

    ```bash
    python scripts/prepare_mydata.py --destination_path data/mydata/
    ```

5. Run `finetune_full.py` by passing in the location of your data (and optionally other parameters):
   
    ```bash
    python finetune_full.py --data_dir data/mydata/ --out_dir out/myexperiment
    ```


## Troubleshooting

If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
