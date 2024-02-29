## Download [StableLM](https://github.com/Stability-AI/StableLM) weights

StableLM is a family of generative language models trained by StabilityAI.

To see all the available checkpoints for StableLM, run:

```bash
python scripts/download.py | grep stablelm
```

which will print

```text
stabilityai/stablelm-3b-4e1t
stabilityai/stablelm-zephyr-3b
```

### StableLM-3B-4E1T

StableLM-3B-4E1T is a 3 billion (3B) parameter language model pre-trained under the multi-epoch regime to study the impact of repeated tokens on downstream performance.

Building on past achievements, StabilityAI underwent training on 1 trillion tokens for 4 epochs, as recommended by Muennighoff et al. (2023) in their study "Scaling Data-Constrained Language Models." They noted that training with repeated data over 4 epochs has minimal impact on loss compared to using unique data. Additionally, insights from "Go smol or go home" (De Vries, 2023) guided the choice of token count. The research suggests that a 2.96B model trained on 2.85 trillion tokens can achieve a loss similar to a compute-optimized 9.87B language model.
More info can be found on [GitHub](https://github.com/Stability-AI/StableLM?tab=readme-ov-file#stablelm-3b-4e1t).

### StableLM Zephyr 3B

Lightweight LLM, preference tuned for instruction following and Q&A-type tasks. This model is an extension of the pre-existing StableLM 3B-4e1t model and is inspired by the Zephyr 7B model from HuggingFace. With StableLM Zephyr's 3 billion parameters, this model efficiently caters to a wide range of text generation needs, from simple queries to complex instructional contexts on edge devices.
More details can be found in the [announcement](https://stability.ai/news/stablelm-zephyr-3b-stability-llm).

### Usage

In order to use a specific StableLM checkpoint, for instance [StableLM Zephyr 3B](https://huggingface.co/stabilityai/stablelm-zephyr-3b), download the weights and convert the checkpoint to the lit-gpt format:

```bash
pip install 'huggingface_hub[hf_transfer] @ git+https://github.com/huggingface/huggingface_hub'

export repo_id=stabilityai/stablelm-zephyr-3b
python scripts/download.py --repo_id $repo_id --from_safetensors=True
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$repo_id
```

By default, the `convert_hf_checkpoint` step will use the data type of the HF checkpoint's parameters. In cases where RAM
or disk size is constrained, it might be useful to pass `--dtype bfloat16` to convert all parameters into this smaller precision before continuing.

You're done! To execute the model just run:

```bash
pip install tokenizers

python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b
```

Or you can run the model in an interactive mode:

```bash
python chat/base.py --checkpoint_dir checkpoints/$repo_id
```
