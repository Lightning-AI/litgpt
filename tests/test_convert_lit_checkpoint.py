import torch
import json
import os
import pytest


@pytest.mark.parametrize("model_size", ["7b", "40b"])
def test_falcon_conversion(model_size):
    chkptpath = f"checkpoints/tiiuae/falcon-{model_size}/lit_model_finetuned.bin"

    if not os.path.isfile(chkptpath):
        pytest.skip(f"{chkptpath} not found")

    lit_model = torch.load(chkptpath)
    lit_weight_keys = [k for k in lit_model.keys()]

    hf_keys_path = "checkpoints/tiiuae/falcon-7b/pytorch_model.bin.index.json"
    with open(hf_keys_path, "r") as json_file:
        hf_md = json.load(json_file)
    hf_weight_keys = list(hf_md["weight_map"])

    assert all(litkey in hf_weight_keys for litkey in lit_weight_keys)
