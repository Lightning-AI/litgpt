# adapted from karpathy/minGPT

import torch
from models.llama import LLAMA, Tokenizer
from models.llama.model import LLAMA_CONFIG_DICT
from lightning import seed_everything


def generate(
    prompt, num_samples=10, steps=20, do_sample=True, top_k=200, temperature=0.8
):
    device = torch.device("cuda")
    checkpoint = torch.load("./data/checkpoints/llama/converted_meta/7B/state_dict.pt")

    with device:
        model = LLAMA(LLAMA_CONFIG_DICT["7B"])
        model.load_state_dict(checkpoint)

    tokenizer = Tokenizer("./data/checkpoints/llama/converted_meta/tokenizer.model")

    model.to(device)
    model.eval()

    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False).to(device)
    encoded_prompt = encoded_prompt[None, :]

    for k in range(num_samples):
        y = model.generate(encoded_prompt, steps, temperature=temperature, top_k=top_k)
        print(tokenizer.decode(y[0]))


if __name__ == "__main__":
    seed_everything(12334)
    generate(prompt="Hello, my name is", num_samples=10, steps=20)
