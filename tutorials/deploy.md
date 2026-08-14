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
## Serve an LLM with OpenAI-compatible API

LitGPT provides OpenAI-compatible endpoints that allow you to use the OpenAI SDK or any OpenAI-compatible client to interact with your models. This is useful for integrating LitGPT into existing applications that use the OpenAI API.

&nbsp;
### Step 1: Start the server with OpenAI specification

```bash
# 1) Download a pretrained model (alternatively, use your own finetuned model)
litgpt download HuggingFaceTB/SmolLM2-135M-Instruct

# 2) Start the server with OpenAI-compatible endpoints
litgpt serve HuggingFaceTB/SmolLM2-135M-Instruct --openai_spec true
```

> [!TIP]
> The `--openai_spec true` flag enables OpenAI-compatible endpoints at `/v1/chat/completions` instead of the default `/predict` endpoint.

&nbsp;
### Step 2: Query using OpenAI-compatible endpoints

You can now send requests to the OpenAI-compatible endpoint using curl:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM2-135M-Instruct",
    "messages": [{"role": "user", "content": "Hello! How are you?"}]
  }'
```

Or use the OpenAI Python SDK:

```python
from openai import OpenAI

# Configure the client to use your local LitGPT server
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"  # LitGPT doesn't require authentication by default
)

response = client.chat.completions.create(
    model="SmolLM2-135M-Instruct",
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ]
)

print(response.choices[0].message.content)
```

&nbsp;
## Serve an LLM UI with Chainlit

If you are interested in developing a simple ChatGPT-like UI prototype, see the Chainlit tutorial in the following Studio:

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/chatgpt-like-llm-uis-via-chainlit">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>
