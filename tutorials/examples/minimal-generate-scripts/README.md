## Minimal LitGPT Generate Examples in Python



The scripts in this folder provide minimal examples showing how to use LitGPT from within Python without the CLI. 

- `generate.py` is a minimal script that uses the `main` function from LitGPT's `generate` utilities
- `generate-step-by-step.py` is a lower-level script using LitGPT utility functions directly instead of relying on the `main` function menntioned above.

Assuming you downloaded the checkpoint files via 

```bash
litgpt download EleutherAI/pythia-1b
```

you can run the scripts as follows:


```bash
python generate-step-by-step.py
```

or

```bash
python generate.py
```



