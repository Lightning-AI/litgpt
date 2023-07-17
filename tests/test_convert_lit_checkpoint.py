import torch
import json

if __name__ == "__main__":
    lit_hf_ckpt = "checkpoints/tiiuae/falcon-7b/lit_hf_model.bin"
    lit_hf_model = torch.load(lit_hf_ckpt)

    lit_hf_weights_keys = [k for k in lit_hf_model.keys()]

    hf_keys_path = "checkpoints/tiiuae/falcon-7b/pytorch_model.bin.index.json"

    with open(hf_keys_path, "r") as json_file:
        hf_md = json.load(json_file)

    hf_weight_map = hf_md["weight_map"]

    assert all(
        litkey in list(hf_weight_map.keys()) for litkey in lit_hf_weights_keys
    ), "does not match"

    print("passed")
