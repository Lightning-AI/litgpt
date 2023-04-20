import gc
import json
import shutil
import sys
from pathlib import Path

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice


@torch.no_grad()
def convert_hf_checkpoint(
    *,
    output_dir: Path = Path("checkpoints/lit-llama"),
    ckpt_dir: Path = Path("checkpoints/hf-llama/"),
    model_size: str = "7B",
    dtype: str = "float32",
    verify: bool = False,
) -> None:
    """
    Perform the reverse operation of: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
    """
    output_dir = output_dir / model_size
    ckpt_dir = ckpt_dir / model_size
    output_dir.mkdir(parents=True, exist_ok=True)

    # the tokenizer is the same for all model sizes, so we store it in the parent dir
    shutil.copy(ckpt_dir / "tokenizer.model", output_dir.parent)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    print("Initializing lit-llama")
    config = LLaMAConfig.from_name(model_size)

    with EmptyInitOnDevice(device="cpu", dtype=dtype):
        model = LLaMA(config)

    qkv_size = model.transformer.h[0].attn.c_attn.weight.shape[0] // 3

    # initialize a new empty state dict to hold our new weights
    sd = model.state_dict()

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = ckpt_dir / "pytorch_model.bin.index.json"
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

        hf_weights = torch.load(ckpt_dir / bin_file, map_location="cpu")

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

    print(f"Saving to disk at {output_dir}")
    torch.save(model.state_dict(), output_dir / "lit-llama.pth")

    if verify:
        try:
            from transformers import LlamaForCausalLM
        except ImportError:
            raise ImportError("verify=True requires transformers to be installed, please `pip install transformers`")
        print("Verifying...")

        token_sample = torch.randint(0, config.vocab_size, size=(1, config.block_size), dtype=torch.int64)
        out = model(token_sample)
        del model
        gc.collect()

        print("Loading original model for comparison")
        model_hf = LlamaForCausalLM.from_pretrained(ckpt_dir)
        out_hf = model_hf(token_sample)["logits"]

        print("Comparing outputs")
        assert out.device.type == out_hf.device.type
        assert out.dtype == out_hf.dtype
        assert torch.testing.assert_close(out, out_hf)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint)

