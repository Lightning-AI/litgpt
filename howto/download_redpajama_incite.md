## Download [RedPajama-INCITE](https://www.together.xyz/blog/redpajama-models-v1) weights

togethercomputer's RedPajama-INCITE family of models were trained over the [RedPajama v1](https://www.together.xyz/blog/redpajama) dataset, with the same architecture as the popular [Pythia](download_pythia.md) model suite.
It includes a base model, a chat fine-tuned model, and an instruction tuned model of 3B and 7B sizes.

The process to use the RedPajama-INCITE weights is the same as described in our [StableLM guide](download_stablelm.md).

For instance, to run inference with the [RedPajama-INCITE-Base-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1) checkpoint:

```bash
python scripts/download.py togethercomputer/RedPajama-INCITE-Base-3B-v1

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/togethercomputer/RedPajama-INCITE-Base-3B-v1

python generate.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/togethercomputer/RedPajama-INCITE-Base-3B-v1
```
