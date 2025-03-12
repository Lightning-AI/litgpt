## Minimal PyTorch Lightning Trainer Example



The script in this folder provides minimal examples showing how to train a LitGPT model using LitGPT's `GPT` class with the [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) Trainer.

You can run the scripts as follows:

&nbsp
## Small 160M model:

```bash
# Download the Pythia model
litgpt download EleutherAI/pythia-160m

python litgpt_ptl_small.py
```

&nbsp
## Medium-sized 8B model:

```bash
# Download the Llama 3.1 model
litgpt download meta-llama/Meta-Llama-3.1-8B --access_token hf_...

python litgpt_ptl_medium.py
```
