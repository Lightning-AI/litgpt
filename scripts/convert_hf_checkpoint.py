import gc
import json
import shutil
import sys
from pathlib import Path

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_stablelm.model import StableLM
from lit_stablelm.utils import EmptyInitOnDevice


def copy_weights(state_dict, hf_weights, dtype=torch.float32):
    weight_map = {
        "gpt_neox.embed_in.weight": "transformer.wte.weight",
        'gpt_neox.layers.{}.input_layernorm.bias': "transformer.h.{}.norm_1.bias",
        'gpt_neox.layers.{}.input_layernorm.weight': "transformer.h.{}.norm_1.weight",
        'gpt_neox.layers.{}.attention.query_key_value.bias': "transformer.h.{}.attn.attn.bias",
        'gpt_neox.layers.{}.attention.query_key_value.weight': "transformer.h.{}.attn.attn.weight",
        'gpt_neox.layers.{}.attention.dense.bias': "transformer.h.{}.attn.proj.bias",
        'gpt_neox.layers.{}.attention.dense.weight': "transformer.h.{}.attn.proj.weight",
        'gpt_neox.layers.{}.attention.rotary_emb.inv_freq': None,
        'gpt_neox.layers.{}.attention.bias': None,
        'gpt_neox.layers.{}.attention.masked_bias': None,
        'gpt_neox.layers.{}.post_attention_layernorm.bias': "transformer.h.{}.norm_2.bias",
        'gpt_neox.layers.{}.post_attention_layernorm.weight': "transformer.h.{}.norm_2.weight",
        'gpt_neox.layers.{}.mlp.dense_h_to_4h.bias': "transformer.h.{}.mlp.fc.bias",
        'gpt_neox.layers.{}.mlp.dense_h_to_4h.weight': "transformer.h.{}.mlp.fc.weight",
        'gpt_neox.layers.{}.mlp.dense_4h_to_h.bias': "transformer.h.{}.mlp.proj.bias",
        'gpt_neox.layers.{}.mlp.dense_4h_to_h.weight': "transformer.h.{}.mlp.proj.weight",
        'gpt_neox.final_layer_norm.bias': "transformer.ln_f.bias",
        'gpt_neox.final_layer_norm.weight': "transformer.ln_f.weight",
        'embed_out.weight': "lm_head.weight",
    }

    for name, param in hf_weights.items():
        param = param.to(dtype=dtype)
        if "gpt_neox.layers" in name:
            split = name.split(".")
            block_id = int(split[2])
            split[2] = "{}"
            from_name = ".".join(split)
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(block_id)
        else:
            to_name = weight_map[name]
        print(f"{name} {tuple(param.shape)} ⟶ {to_name} {tuple(state_dict[to_name].shape)}")
        state_dict[to_name].copy_(param)


@torch.no_grad()
def convert_hf_checkpoint(
    *,
    output_dir: Path = Path("checkpoints/lit-stablelm"),
    ckpt_dir: Path = Path("checkpoints/hf-stablelm/"),
    model_size: str = "7B",
    dtype: str = "float32",
    verify: bool = False,
) -> None:
    output_dir = output_dir / model_size
    ckpt_dir = ckpt_dir / model_size
    output_dir.mkdir(parents=True, exist_ok=True)

    # the tokenizer is the same for all model sizes, so we store it in the parent dir
    shutil.copy(ckpt_dir / "tokenizer.json", output_dir.parent)
    shutil.copy(ckpt_dir / "tokenizer_config.json", output_dir.parent)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    print("Initializing lit-stablelm")
    with EmptyInitOnDevice(device="cpu", dtype=dtype):
        model = StableLM.from_name(model_size)

    with open(output_dir / "config.json", "w") as json_config:
        json.dump(model.config.__dict__, json_config)

    # initialize a new empty state dict to hold our new weights
    sd = model.state_dict()

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = ckpt_dir / "pytorch_model.bin.index.json"
    with open(pytorch_bin_map_json_path) as json_map:
        bin_index = json.load(json_map)

    bin_files = sorted(set(el for el in bin_index["weight_map"].values()))

    for bin_file in bin_files:
        print("Processing", bin_file)
        hf_weights = torch.load(ckpt_dir / bin_file, map_location="cpu")
        copy_weights(sd, hf_weights, dtype=dtype)
        del hf_weights
        gc.collect()

    print(f"Saving to disk at {output_dir}")
    torch.save(model.state_dict(), output_dir / "lit-stablelm.pth")

    if verify:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("verify=True requires transformers to be installed, please `pip install transformers`")
        print("Verifying...")

        config = model.config
        token_sample = torch.randint(0, config.vocab_size, size=(1, config.block_size), dtype=torch.int64)
        out = model(token_sample)
        del model
        gc.collect()

        print("Loading original model for comparison")
        model_hf = AutoModelForCausalLM.from_pretrained(ckpt_dir)
        out_hf = model_hf(token_sample)["logits"]

        print("Comparing outputs")
        assert out.device.type == out_hf.device.type
        assert out.dtype == out_hf.dtype
        assert torch.testing.assert_close(out, out_hf)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint)

