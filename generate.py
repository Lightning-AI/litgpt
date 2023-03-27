import os
import time
import torch
from tokenizer import Tokenizer
import lightning as L
from quantization.bnb import quantize as quantize_model
import sys


@torch.no_grad()
def generate(model, idx, max_new_tokens, max_seq_length, temperature=1.0, top_k=None):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (B, T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    B, T = idx.shape
    T_new = T + max_new_tokens
    empty = torch.empty(B, T_new, dtype=idx.dtype, device=idx.device)
    empty[:, :T] = idx
    idx = empty

    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:, :t]
        # if the sequence context is growing too long we must crop it at max_seq_length
        idx_cond = idx_cond if T <= max_seq_length else idx_cond[:, -max_seq_length:]

        # forward
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # concatenate the new column
        idx[:, t] = idx_next

    return idx


def get_model(original: bool = False):
    if original:
        try:
            from original_model import Transformer, ModelArgs
        except ModuleNotFoundError:
            from scripts.download import download_original

            download_original(os.path.dirname(__file__))

            from original_model import Transformer, ModelArgs

        config = ModelArgs(dim=4096, n_layers=32, n_heads=32, vocab_size=32000, max_batch_size=1)  # 7B config
        return Transformer(config), config.max_seq_len
    else:
        from model import LLaMA, LLaMAConfig

        config = LLaMAConfig()  # 7B default
        return LLaMA(config), config.block_size


def main(
    prompt: str = "Hello, my name is",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
    # compilation fails as it does not support torch.complex64 for RoPE
    # compile: bool = False,
    accelerator: str = "auto",
    checkpoint_path: str = "/srv/data/checkpoints/llama/converted_nano/7B/state_dict.pth",
    tokenizer_path: str = "/srv/data/checkpoints/llama/converted_nano/tokenizer.model",
    original_model: bool = False,
    quantize: bool = False,
):
    """
    Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        # compile: Whether to compile the model.
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        original_model: Whether to use the original LLaMA model from Meta.
        quantize: Whether to quantize the model using the `LLM.int8()` method
    """
    assert os.path.isfile(checkpoint_path)
    assert os.path.isfile(tokenizer_path)

    fabric = L.Fabric(accelerator=accelerator, devices=1)

    if quantize:
        print("Running quantization. This may take a minute ...")
        # TODO: Initializing the model directly on the device does not work with quantization
        model, max_seq_length = get_model(original_model)

        # The output layer can be sensitive to quantization, we keep it in default precision
        model = quantize_model(model, skip=("lm_head", "output", ))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint, strict=(not original_model))
    else:
        with fabric.device:
            model, max_seq_length = get_model(original_model)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint, strict=(not original_model))

    model.eval()
    
    # if compile:
    #     model = torch.compile(model)

    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False).to(fabric.device)
    encoded_prompt = encoded_prompt[None, :]

    L.seed_everything(1234)
    t0 = time.time()
    for _ in range(num_samples):
        y = generate(
            model, encoded_prompt, max_new_tokens, max_seq_length, temperature=temperature, top_k=top_k
        )
        print(tokenizer.decode(y[0]))

    print(f"Time for inference: {time.time() - t0:.02f} seconds", file=sys.stderr)
    print(f"Memory used (GB): {torch.cuda.max_memory_reserved() / 1e9:.02f}", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision('high')
    CLI(main)
