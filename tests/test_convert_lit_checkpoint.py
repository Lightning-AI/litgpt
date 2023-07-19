import torch
import json
import os
import pytest


@pytest.mark.parametrize("model_size", ["7b", "40b"])
def test_copy_weights_falcon(model_size):
    reconstructed_model_path = f"checkpoints/tiiuae/falcon-{model_size}/lit_model_finetuned.bin"
    bin_model_path = f"checkpoints/tiiuae/falcon-{model_size}/pytorch_model.bin.index.json"

    if not os.path.isfile(reconstructed_model_path):
        pytest.skip(f"{reconstructed_model_path} not found")

    lit_model = torch.load(reconstructed_model_path)
    lit_keys = [k for k in lit_model.keys()]

    with open(bin_model_path, "r") as json_file:
        hf_md = json.load(json_file)
    torch_keys = list(hf_md["weight_map"])

    assert all(litkey in torch_keys for litkey in lit_keys)
    assert len(lit_keys) == len(torch_keys)


def test_copy_weights_hf_llama(tmp_path):
    reconstructed_model_path = "checkpoints/openlm-research/open_llama_3b/lit_model_finetuned.bin"
    bin_model_path = "checkpoints/openlm-research/open_llama_3b/pytorch_model.bin"

    if not os.path.isfile(reconstructed_model_path):
        pytest.skip(f"{reconstructed_model_path} not found")

    lit_model = torch.load(reconstructed_model_path)
    lit_keys = [k for k in lit_model.keys()]

    torch_model = torch.load(bin_model_path)
    torch_keys = [k for k in torch_model.keys()]

    assert all(litkey in torch_keys for litkey in lit_keys)
    assert len(lit_keys) == len(torch_keys)


def test_copy_weights_gpt_neox(tmp_path):
    reconstructed_model_path = "checkpoints/EleutherAI/pythia-1b/lit_model_finetuned.bin"
    bin_model_path = "checkpoints/EleutherAI/pythia-1b/pytorch_model.bin"

    if not os.path.isfile(reconstructed_model_path):
        pytest.skip(f"{reconstructed_model_path} not found")

    lit_model = torch.load(reconstructed_model_path)
    lit_keys = [k for k in lit_model.keys()]

    torch_model = torch.load(bin_model_path)
    torch_keys = [k for k in torch_model.keys()]

    assert all(litkey in torch_keys for litkey in lit_keys)
    assert len(lit_keys) == len(torch_keys)
