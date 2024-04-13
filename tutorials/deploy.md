# Deploy and Serve LLMs

This document shows how you can serve a LitGPT for deployment. 

&nbsp;
## Serve an LLM with LitServe

This section illustrates how we can set up an inference server for a phi-2 LLM using [LitServe](https://github.com/Lightning-AI/litserve).

[LitServe](https://github.com/Lightning-AI/litserve) is an inference server for AI/ML models that is minimal and highly scalable.

You can install LitServe as follows:

```bash
pip install litserve
```

&nbsp;
### Step 1: Create a server.py file

First, copy the following code into a file called `server.py`:

```python
from pathlib import Path

import lightning as L
import torch
from litserve import LitAPI, LitServer

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.generate.base import generate
from litgpt.prompts import load_prompt_style, has_prompt_style, PromptStyle
from litgpt.scripts.download import download_from_hub
from litgpt.utils import load_checkpoint


# DEFINE YOUR MODEL API
class SimpleLitAPI(LitAPI):

    def setup(self, device):
        # Setup the model so it can be called in `predict`.
        repo_id = "microsoft/phi-2"
        checkpoint_dir = Path(f"checkpoints/{repo_id}")

        if not checkpoint_dir.exists():
            download_from_hub(repo_id=repo_id)

        config = Config.from_file(checkpoint_dir / "model_config.yaml")

        device = torch.device(device)
        torch.set_float32_matmul_precision("high")
        fabric = L.Fabric(accelerator=device.type, devices=[device.index], precision="bf16-true")

        checkpoint_path = checkpoint_dir / "lit_model.pth"

        self.tokenizer = Tokenizer(checkpoint_dir)
        self.prompt_style = (
            load_prompt_style(checkpoint_dir) if has_prompt_style(checkpoint_dir) else PromptStyle.from_config(config)
        )

        with fabric.init_module(empty_init=True):
            model = GPT(config)
        with fabric.init_tensor():
            # enable the kv cache
            model.set_kv_cache(batch_size=1)
        model.eval()

        self.model = fabric.setup_module(model)

        load_checkpoint(fabric, self.model, checkpoint_path)

        self.device = fabric.device

    def decode_request(self, request):
        # Convert the request payload to your model input.
        prompt = request["prompt"]
        prompt = self.prompt_style.apply(prompt)
        encoded = self.tokenizer.encode(prompt, device=self.device)
        return encoded

    def predict(self, inputs):
        # Run the model on the input and return the output.
        prompt_length = inputs.size(0)
        max_returned_tokens = prompt_length + 30

        y = generate(self.model, inputs, max_returned_tokens, temperature=0.8, top_k=200, eos_id=self.tokenizer.eos_id)

        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        return y

    def encode_response(self, output):
        # Convert the model output to a response payload.
        decoded_output = self.tokenizer.decode(output)
        return {"output": decoded_output}


# START THE SERVER
if __name__ == "__main__":
    server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=1)
    server.run(port=8000)
```

&nbsp;
## Step 2: Start the inference server

After you saved the code from step 1 in a `server.py` file, start the inference server from your command line terminal:

```bash
python server.py
```

&nbsp;
## Step 3: Query the inference server

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
