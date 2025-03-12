# Serve and Deploy LLMs

This document shows how you can serve a LitGPT for deployment.


&nbsp;
## Serve an LLM with LitServe

This section illustrates how we can set up an inference server for a phi-2 LLM using `litgpt serve` that is minimal and highly scalable.


&nbsp;
### Step 1: Start the inference server


```bash
# 1) Download a pretrained model (alternatively, use your own finetuned model)
litgpt download microsoft/phi-2

# 2) Start the server
litgpt serve microsoft/phi-2
```

> [!TIP]
> Use `litgpt serve --help` to display additional options, including the port, devices, LLM temperature setting, and more.


&nbsp;
### Step 2: Query the inference server

You can now send requests to the inference server you started in step 2. For example, in a new Python session, we can send requests to the inference server as follows:


```python
import requests, json

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"prompt": "Fix typos in the following sentence: Example input"}
)

print(response.json()["output"])
```

Executing the code above prints the following output:

```
Example input.
```

&nbsp;
### Optional: Use the streaming mode

The 2-step procedure described above returns the complete response all at once. If you want to stream the response on a token-by-token basis, start the server with the streaming option enabled:

```bash
litgpt serve microsoft/phi-2 --stream true
```

Then, use the following updated code to query the inference server:

```python
import requests, json

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"prompt": "Fix typos in the following sentence: Example input"},
    stream=True
)

# stream the response
for line in response.iter_lines(decode_unicode=True):
    if line:
        print(json.loads(line)["output"], end="")
```

```
Sure, here is the corrected sentence:

Example input
```

&nbsp;
## Serve an LLM UI with Chainlit

If you are interested in developing a simple ChatGPT-like UI prototype, see the Chainlit tutorial in the following Studio:

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/chatgpt-like-llm-uis-via-chainlit">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>
