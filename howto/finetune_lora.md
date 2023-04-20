# Finetuning with LoRA

[Low-rank adaption (LoRA)](https://arxiv.org/abs/2106.09685) is a technique to approximate the update to the linear layers in a LLM with a low-rank matrix factorization. This significantly reduces the number of trainable parameters and speeds up training with little impact on the final performance of the model.
We demonstrate this method by instruction-finetuning LLaMA 7B on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset on a **single GTX 3090 (24GB) GPU**.

## Preparation

The steps here only need to be done once:

1. Follow the instructions in the [README](README.md) to install the dependencies.
2. Download and convert the weights and save them in the `./checkpoints` folder as described [here](download_weights.md).
3. Download the data and generate the instruction tuning dataset:

   ```bash
   python scripts/prepare_alpaca.py
   ```

## Running the finetuning

```bash
python finetune_lora.py
```

The finetuning requires at least one GPU with ~24 GB memory (GTX 3090).

This script will save checkpoints periodically to the folder `out/`.

> **Note**
> All scripts support argument [customization](customize_paths.md)


## Test the model

You can test the finetuned model with your own instructions by running:

```bash
python generate_lora.py --prompt "Recommend a movie to watch on the weekend."
```
Output:
```
I would recommend the movie The Martian (2015). It is a sci-fi movie starring Matt Damon that follows the story of...
```

If your GPU supports `bfloat16`, you can additionally pass `--dtype bfloat16` to bring the memory consumption down to ~14 GB.

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

5. Run `finetune_lora.py` by passing in the location of your data (and optionally other parameters):
    
    ```bash
    python finetune_lora.py --data_dir data/mydata/ --out_dir out/myexperiment
    ```


## Troubleshooting

If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
