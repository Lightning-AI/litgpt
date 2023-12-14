# Inference

We demonstrate how to run inference (next token prediction) with the GPT base model in the [`generate.py`](generate.py) script:

```bash
python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

Output:

```text
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

## Run a large model on one smaller device

Check out our [quantization tutorial](quantize.md).

## Run a large model on multiple smaller devices

You can also use the `generate/sequentially.py` script to leverage multiple devices to perform inference.
This will allow you to run models that wouldn't fit in a single card by partitioning the weights across all your devices and running the layers sequentially.

For instance, `falcon-180b` would require ~360 GB of GPU memory to load on a single device, plus the memory from activations.
We can instead run it on TODO A100 40GB GPUs instead of 10 GPUs:

```shell
CUDA_VISIBLE_DEVICES=TODO python generate/sequentially.py \
  --checkpoint_dir checkpoints/tiiuae/falcon-180b \
  --max_new_tokens TODO
```

Taking ~25 GB of memory, and run at 2.5 tokens/sec.

The script also supports quantization, using 4-bit precision, we can use 4 GPUs

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python generate/sequentially.py \
  --checkpoint_dir checkpoints/tiiuae/falcon-180b \
  --max_new_tokens TODO \
  --quantize bnb.nf4
```

Which will take ~25 GB of memory, and run at 2.5 tokens/sec.

Smaller devices like 3090s or A10Gs (24 GB) can also fit it with this technique.