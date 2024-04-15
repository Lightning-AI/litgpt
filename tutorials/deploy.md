# Serve and Deploy LLMs

This document shows how you can serve a LitGPT for deployment. 

&nbsp;
## Serve an LLM

This section illustrates how we can set up an inference server for a phi-2 LLM using `litgpt serve` that is minimal and highly scalable.


&nbsp;
## Step 1: Start the inference server


```bash
# 1) Download a pretrained model (alternatively, use your own finetuned model)
litgpt download --repo_id microsoft/phi-2

# 2) Start the server
litgpt serve --checkpoint_dir checkpoints/microsoft/phi-2
```

> [!TIP]
> Use `litgpt serve --help` to display additional options, including the port, devices, LLM temperature setting, and more.


&nbsp;
## Step 2: Query the inference server

You can now send requests to the inference server you started in step 2. For example, in a new Python session, we can send requests to the inference server as follows:


```python
import requests, json

response = requests.post(
    "http://127.0.0.1:8000/predict", 
    json={"prompt": "Fix typos in the following sentence: Exampel input"}
)

decoded_string = response.content.decode("utf-8")
output_str = json.loads(decoded_string)["output"]
print(output_str)
```

Executing the code above prints the following output:

```
Instruct: Fix typos in the following sentence: Exampel input
Output: Example input.
```
