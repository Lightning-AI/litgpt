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
    compile: bool = False,
):
    """
    Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        steps: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random samples.
        compile: Whether to compile the model.
    """
    L.seed_everything(1234)

    device = torch.device("cuda")
    checkpoint = torch.load("/srv/data/checkpoints/llama/converted_meta/7B/state_dict.pt")

    with device:
        model = LLAMA(LLAMA_CONFIG_DICT["7B"])
        model.load_state_dict(checkpoint)
    model.eval()

    if compile:
        model = torch.compile(model)

    tokenizer = Tokenizer("/srv/data/checkpoints/llama/converted_meta/tokenizer.model")
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False).to(device)
    encoded_prompt = encoded_prompt[None, :]
    for k in range(num_samples):
        y = model.generate(encoded_prompt, steps, temperature=temperature, top_k=top_k)
        print(tokenizer.decode(y[0]))


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI()
