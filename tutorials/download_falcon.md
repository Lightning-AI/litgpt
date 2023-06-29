## Download [Falcon](https://falconllm.tii.ae) weights

UAE's Technology Innovation Institute has open-sourced Falcon LLM.
It is trained on [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) enhanced with curated corpora
 Weights are released under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

The release includes a base model and an instruction tuned model of sizes 7B and 40B.

To see all the available checkpoints for Falcon, run:

```bash
python scripts/download.py | grep falcon
```

which will print

```text
tiiuae/falcon-7b
tiiuae/falcon-7b-instruct
tiiuae/falcon-40b
tiiuae/falcon-40b-instruct
```

In order to use a specific Falcon checkpoint, for instance [falcon-7b](https://huggingface.co/tiiuae/falcon-7b), download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install huggingface_hub

python scripts/download.py --repo_id tiiuae/falcon-7b

python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

You're done! To execute the model just run:

```bash
python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lightning-AI/lit-gpt/blob/main/notebooks/falcon-inference.ipynb)
