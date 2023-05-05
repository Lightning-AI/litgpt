# Inference

We demonstrate how to run inference (next token prediction) with the LLaMA base model in the [`generate.py`](generate.py) script:

```bash
python generate.py --prompt "Hello, my name is"
```
Output:
```
Hello, my name is Levi Durrer, I'm an Austrian journalist - Chairman of the Press Blair Party, with 37 years in the Press Blair International, and two years in the Spectre of Austerity for the other. I'm crossing my fingers that you will feel
```

The script assumes you have downloaded and converted the weights and saved them in the `./checkpoints` folder as described [here](download_weights.md).

> **Note**
> All scripts support argument [customization](customize_paths.md)

This will run the 3B pre-trained model and require ~7 GB of GPU memory using the `bfloat16` datatype.


