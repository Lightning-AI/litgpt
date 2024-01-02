## Download [Qwen](https://github.com/QwenLM/Qwen) weights

Qwen-7B is the 7B-parameter version of the large language model series, Qwen (abbr. Tongyi Qianwen), proposed by Alibaba Cloud. 

For more info on the models, please see the [Qwen repository](https://github.com/QwenLM/Qwen).

To see all the available checkpoints for Qwen, run:

```bash
python scripts/download.py | grep Qwen
```

which will print

```text
Qwen/Qwen-7B
```

In order to use a specific Qwen checkpoint, for instance [Qwen-7B](https://huggingface.co/Qwen/Qwen-7B), download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id Qwen/Qwen-7B --from_safetensors true

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/Qwen/Qwen-7B
```

By default, the convert_hf_checkpoint step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install tiktoken

python generate/base.py --prompt "中国的首都是" --checkpoint_dir checkpoints/Qwen/Qwen-7B/
```
