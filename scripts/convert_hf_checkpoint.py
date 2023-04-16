import gc
import os
import json
from pathlib import Path
import sys
from typing import Optional

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice


@torch.no_grad()
def convert_hf_checkpoint(
    model_size: str = "7B",
    hf_checkpoint_path: Path = Path("checkpoints/llama-7b-hf"),
    lit_checkpoint: Path = Path("checkpoints/lit-llama.pth"),
    dtype: str = "float32",
    verify: bool = False,
) -> None:
    """
    Perform the reverse operation of: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
    """

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    print("Initializing lit-llama")
    config = LLaMAConfig.from_name(model_size)

    with EmptyInitOnDevice(device="cpu", dtype=dtype):
        model = LLaMA(config)

    sd = model.state_dict()
    qkv_size = model.transformer.h[0].attn.c_attn.weight.shape[0] // 3

    # initialize a new empty state dict to hold our new weights
    sd = model.state_dict()

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = os.path.join(hf_checkpoint_path, "pytorch_model.bin.index.json")
    with open(pytorch_bin_map_json_path) as json_map:
        bin_index = json.load(json_map)

    bin_files = set(el for el in bin_index["weight_map"].values())

    def permute(w):
        dim = config.n_embd
        return (
            w.view(config.n_head, 2, dim // config.n_head // 2, dim)
            .transpose(1, 2)
            .reshape(dim, dim)
        )

    weight_map = {
        "self_attn.o_proj.weight": "attn.c_proj.weight",
        "self_attn.q_proj.weight": "attn.c_attn.weight",
        "self_attn.k_proj.weight": "attn.c_attn.weight",
        "self_attn.v_proj.weight": "attn.c_attn.weight",
        "mlp.gate_proj.weight": "mlp.c_fc1.weight",
        "mlp.up_proj.weight": "mlp.c_fc2.weight",
        "mlp.down_proj.weight": "mlp.c_proj.weight",
        "input_layernorm.weight": "rms_1.scale",
        "post_attention_layernorm.weight": "rms_2.scale",
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.norm.weight": "transformer.ln_f.scale",
        "lm_head.weight": "lm_head.weight"
    }

    for bin_file in bin_files:
        print("Processing", bin_file)

        hf_weights = torch.load(os.path.join(hf_checkpoint_path, bin_file), map_location="cpu")

        for name, param in hf_weights.items():
            param = param.to(dtype=dtype)
            if "rotary_emb.inv_freq" in name:
                continue
            if "model.layers" in name:
                block_id = int(name.split(".")[2])
                from_name = ".".join(name.split(".")[3:])
                to_name = weight_map[from_name]

                if "q_proj" in name:
                    sd[f"transformer.h.{block_id}.{to_name}"][:qkv_size] = permute(param)
                elif "k_proj" in name:
                    sd[f"transformer.h.{block_id}.{to_name}"][qkv_size:-qkv_size] = permute(param)
                elif "v_proj" in name:
                    sd[f"transformer.h.{block_id}.{to_name}"][-qkv_size:] = param
                else:
                    sd[f"transformer.h.{block_id}.{to_name}"].copy_(param)
            else:
                sd[weight_map[name]].copy_(param)

        del hf_weights
        gc.collect()

    print(f"Saving to disk at {lit_checkpoint}")
    torch.save(model.state_dict(), lit_checkpoint)

    if verify:
        print("Verifying...")

        token_sample = torch.randint(
            0, config.vocab_size, size=(1, config.block_size), dtype=torch.int64
        )

        out = model(token_sample)

        del model
        gc.collect()

        print("Loading original model for comparison.")

        try:
            from transformers import LlamaForCausalLM
        except ImportError as e:
            print("verify=True requires transformers to be installed, please `pip install transformers`")

        model_hf = LlamaForCausalLM.from_pretrained(hf_checkpoint_path)

        out_hf = model_hf(token_sample)

        print("Comparing outputs")
        assert torch.allclose(out, out_hf["logits"])


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint)

