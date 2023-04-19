# Inference

We demonstrate how to run inference (next token prediction) with the LLaMA base model in the [`generate.py`](generate.py) script:

```bash
python generate.py --prompt "Hello, my name is"
```
Output:
```
Hello my name is TJ. I have a passion for the outdoors, love hiking and exploring. I also enjoy traveling and learning new things. I especially enjoy long walks, good conversation and a friendly smile.
```

The script assumes you have downloaded and converted the weights and saved them in the `./checkpoints` folder as described [here](download_weights.md). If you have the weights stored elsewhere, you can pass the weights file in as a parameter:

```bash
python generate.py --prompt "Hello, my name is" \
    --checkpoint_path ./checkpoints/lit-llama/7B/state_dict.pth \
    --tokenizer_path ./checkpoints/lit-llama/tokenizer.model \
    --model_size 7B
```

With the default settings, this will run the 7B model and require ~26 GB of GPU memory (A100 GPU).

## Run Lit-LLaMA on consumer devices

On GPUs with `bfloat16` support, the `generate.py` script will automatically convert the weights and consume about ~14 GB.
For GPUs with less memory, or ones that don't support `bfloat16`, enable quantization (`--quantize llm.int8`):

```bash
python generate.py --quantize llm.int8 --prompt "Hello, my name is"
```
This will consume about ~10 GB of GPU memory or ~8 GB if also using `bfloat16`.
See `python generate.py --help` for more options.

You can also use GPTQ-style int4 quantization, but this needs conversions of the weights first:

```bash
python quantize.py --checkpoint_path lit-llama.pth --tokenizer_path tokenizer.model --output_path llama-7b-gptq.4bit.pt --dtype bfloat16  --quantize gptq.int4
```

With the generated quantized checkpoint generation works as usual with `--quantize gptq.int4`, bringing GPU usage to about ~5GB. As only the weights of the Linear layers are quantized, it is useful to use `--dtype bfloat16` even with the quantization enabled.
