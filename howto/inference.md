# Inference

We demonstrate how to run inference (next token prediction) with the LLaMA base model in the [`generate.py`](generate.py) script:

```bash
python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```
Output:
```
Hello, my name is Levi Durrer, I'm an Austrian journalist - Chairman of the Press Blair Party, with 37 years in the Press Blair International, and two years in the Spectre of Austerity for the other. I'm crossing my fingers that you will feel
```

The script assumes you have downloaded and converted the weights as described [here](download_stablelm.md).

This will run the 3B pre-trained model and require ~7 GB of GPU memory using the `bfloat16` datatype.

## Run interactively

You can also chat with the model interactively:

```bash
python chat/base.py --checkpoint_dir checkpoints/stabilityai/stablelm-tuned-alpha-3b
```

This script can work with any checkpoint. For the best chat-like experience, we recommend using it with a checkpoints
fine-tuned for chatting such as `stabilityai/stablelm-tuned-alpha-3b` or `togethercomputer/RedPajama-INCITE-Chat-3B-v1`.
