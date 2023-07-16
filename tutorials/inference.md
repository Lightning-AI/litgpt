# Inference

We demonstrate how to run inference (next token prediction) with the GPT base model in the [`generate.py`](generate.py) script:

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

## Run a large model on one smaller device

Check out our [quantization tutorial](quantize.md).

## Run a large model on multiple smaller devices

You can also use the Fully-Sharded Data Parallel (FSDP) distributed strategy to leverage multiple devices to perform inference. This will allow you to run models that wouldn't fit in a single card by sharding them across several.

For instance, `falcon-40b` would require ~80 GB of GPU memory to run on a single device. We can instead run it on 4 A100 40GB GPUs:

```shell
python generate/base.py --checkpoint_dir checkpoints/tiiuae/falcon-40b --strategy fsdp --devices 4
```

Which will take 32 GB of memory, and run at 0.37 tokens/sec.

Or to reduce the memory requirements even further, you can try using CPU offloading. For that, you will need to manually edit the `cpu_offload=False` parameter in the file and set it to `True`.

Now we can run it on just 2 devices.

```shell
python generate/base.py --checkpoint_dir checkpoints/tiiuae/falcon-40b --strategy fsdp --devices 2
```

taking 13 GB of memory but running at 0.12 tokens/sec on 2 A100 40GB GPUs.
Smaller devices like 3090s (24 GB) can also fit it with this technique.
