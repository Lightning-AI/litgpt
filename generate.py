# adapted from karpathy/minGPT
import torch
from models.llama import LLAMA, Tokenizer
from models.llama.model import LLAMA_CONFIG_DICT
import lightning as L


def generate(
    prompt: str = "Hello, my name is",
    num_samples: int = 1,
    steps: int = 20,
    top_k: int = 200,
    temperature: float = 0.8,
):
    L.seed_everything(1234)

    device = torch.device("cuda")
    checkpoint = torch.load("./data/checkpoints/llama/converted_meta/7B/state_dict.pt")

    with device:
        model = LLAMA(LLAMA_CONFIG_DICT["7B"])
        model.load_state_dict(checkpoint)
    model.eval()

    tokenizer = Tokenizer("./data/checkpoints/llama/converted_meta/tokenizer.model")
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False).to(device)
    encoded_prompt = encoded_prompt[None, :]
    for k in range(num_samples):
        y = model.generate(encoded_prompt, steps, temperature=temperature, top_k=top_k)
        print(tokenizer.decode(y[0]))


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI()
