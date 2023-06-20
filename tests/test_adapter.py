from dataclasses import asdict

import torch
from lightning import Fabric


def test_config_identical():
    import lit_parrot.adapter as parrot_adapter
    import lit_parrot.model as parrot

    name = "pythia-70m"
    base_config = asdict(parrot.Config.from_name(name))
    adapter_config = asdict(parrot_adapter.Config.from_name(name))
    del adapter_config["adapter_prompt_length"]
    del adapter_config["adapter_start_layer"]
    assert adapter_config == base_config

    with Fabric(accelerator="cpu").init_module(empty_init=True):
        base_model = parrot.Parrot.from_name(name)
        adapter_model = parrot_adapter.Parrot.from_name(name)
    assert adapter_model.lm_head.weight.shape == base_model.lm_head.weight.shape


def test_adapter_filter(tmp_path):
    from lit_parrot.adapter import Parrot, adapter_filter

    fabric = Fabric(devices=1)
    model = Parrot.from_name("pythia-70m", n_layer=4)
    save_path = tmp_path / "model.pth"
    fabric.save(save_path, {"model": model}, filter={"model": adapter_filter})
    saved = torch.load(save_path)["model"]

    expected = {
        "transformer.h.2.attn.adapter_wte.weight",
        "transformer.h.2.attn.gating_factor",
        "transformer.h.3.attn.adapter_wte.weight",
        "transformer.h.3.attn.gating_factor",
    }
    assert set(saved) == expected
