# Inference

We demonstrate how to run inference (next token prediction) with the Parrot base model in the [`generate.py`](generate.py) script:

```bash
python generate.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
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
python chat.py --checkpoint_dir checkpoints/stabilityai/stablelm-tuned-alpha-3b
```

This script can work with any checkpoint. For the best chat-like experience, we recommend using it with a checkpoints
fine-tuned for chatting such as `stabilityai/stablelm-tuned-alpha-3b` or `togethercomputer/RedPajama-INCITE-Chat-3B-v1`.

## Run large models on consumer devices

On GPUs with `bfloat16` support, the `generate.py` script will automatically convert the weights and consume less memory.
For large models, GPUs with less memory, or ones that don't support `bfloat16`, enable quantization (`--quantize llm.int8`):

```bash
python generate.py --quantize llm.int8 --prompt "Hello, my name is"
```
See `python generate.py --help` for more options.

You can also use GPTQ-style int4 quantization, but this needs conversions of the weights first:

```bash
python quantize/gptq.py --dtype bfloat16
```

GPTQ-style int4 quantization brings GPU usage down. As only the weights of the Linear layers are quantized, it is useful to also use `--dtype bfloat16` (default) even with the quantization enabled.

With the generated quantized checkpoint generation quantization then works as usual with `--quantize gptq.int4` and the newly generated checkpoint file:

```bash
python generate.py --quantize gptq.int4
```