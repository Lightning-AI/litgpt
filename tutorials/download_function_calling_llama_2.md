## Download [Function Calling Llama 2](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2) weights

Llama-7B with function calling is licensed according to the Meta Community license.

Function calling Llama extends the hugging face Llama 2 models with function calling capabilities.
The model responds with a structured json argument with the function name and arguments.

In order to use the checkpoint, download the weights and convert the checkpoint to the lit-gpt format.

```bash
pip install huggingface_hub

python scripts/download.py --repo_id Trelis/Llama-2-7b-chat-hf-function-calling-v2 --from_safetensors true

python scripts/convert_hf_checkpoint.py --checkpoint_dir Trelis/Llama-2-7b-chat-hf-function-calling-v2
```

By default, the convert_hf_checkpoint step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install sentencepiece

python chat/base.py --checkpoint_dir Trelis/Llama-2-7b-chat-hf-function-calling-v2
```
Is strongly recommended to visit the model [repository](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2) to know how to format the prompt.

The chat script has a generic use case with a single function defined, feel free to play with it to fit your needs, for instance to make HTTP requests with the model outputs.

Have fun!