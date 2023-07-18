import torch
import json

if __name__ == "__main__":
    lit_ckpt = "checkpoints/tiiuae/falcon-7b/lit_model_finetuned.bin"
    lit_model = torch.load(lit_ckpt)

    lit_weight_keys = [k for k in lit_model.keys()]

    hf_keys_path = "checkpoints/tiiuae/falcon-7b/pytorch_model.bin.index.json"

    with open(hf_keys_path, "r") as json_file:
        hf_md = json.load(json_file)

    hf_weight_map = hf_md["weight_map"]

    assert all(
        litkey in list(hf_weight_map.keys()) for litkey in lit_weight_keys
    ), "does not match"

    print("passed")
