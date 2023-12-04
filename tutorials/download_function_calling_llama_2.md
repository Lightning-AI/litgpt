## Download [Function Calling Llama 2](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2) weights

Llama-7B with function calling is licensed according to the Meta Community license.

Function calling Llama extends the hugging face Llama 2 models with function calling capabilities.
The model responds with a structured json argument with the function name and arguments.


In order to use the checkpoint, download the weights and convert the checkpoint to the lit-gpt format.

```bash
pip install huggingface_hub

python scripts/download.py --repo_id Trelis/Llama-2-7b-chat-hf-function-calling-v2 --access_token your_hf_token --from_safetensors true

python scripts/convert_hf_checkpoint.py --checkpoint_dir Trelis/Llama-2-7b-chat-hf-function-calling-v2
```

By default, the convert_hf_checkpoint step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install sentencepiece

python chat/base.py --checkpoint_dir Trelis/Llama-2-7b-chat-hf-function-calling-v2
```
Is strongly recommended to visit the model autor repository to know how to format the prompt. The chat script has a generic use case with a single function defined, feel free to play with it to fit your needs.

```
    # This is an example for how to format a prompt with a system prompt for the model

    b_func, e_func = "<FUNCTIONS>", "</FUNCTIONS>\n\n"
    b_inst, e_inst = "[INST]", "[/INST]"
    b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    function_metadata = {
                        "function": "search_bing",
                        "description": "Search the web for content on Bing. This allows users to search online/the internet/the web for content.",
                        "arguments": [
                            {
                                "name": "query",
                                "type": "string",
                                "description": "The search query string"
                            }
                        ]
                    }
    
    system_prompt = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as"
        "possible. Your only response should be JSON formatted functions"
    )
    function_list = dumps(function_metadata, indent=4).replace('{', '{{').replace('}', '}}') # Have to replace the curly braces to double curly braces to escape them
    
    system_prompt = f"{b_func}{function_list.strip()}{e_func}{b_inst}{b_sys}{system_prompt.strip()}{e_sys}{'{prompt}'}{e_inst}\n\n"
```

Have fun!