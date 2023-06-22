import torch
from lightning import Fabric


def test_config_identical():
    import lit_gpt.adapter as gpt_adapter
    from lit_gpt.adapter_v2 import adapter_v2_linear_with_bias_and_scale
    import lit_gpt.model as gpt

    name = "pythia-70m"
    with Fabric(accelerator="cpu").init_module(empty_init=True):
        base_model = gpt.GPT.from_name(name)
        adapter_model = gpt_adapter.GPT.from_name(name)

        for module in adapter_model.modules():
            if isinstance(module, torch.nn.Linear):
                adapter_v2_linear_with_bias_and_scale(module)

    assert not hasattr(base_model.transformer.h[2].attn.attn, "adapter_bias")
    assert not hasattr(base_model.transformer.h[2].attn.attn, "adapter_scale")
    assert hasattr(adapter_model.transformer.h[2].attn.attn, "adapter_bias")
    assert hasattr(adapter_model.transformer.h[2].attn.attn, "adapter_scale")


def test_adapter_filter(tmp_path):
    from lit_gpt.adapter_v2 import GPT, adapter_filter

    fabric = Fabric(devices=1)
    model = GPT.from_name("pythia-70m", n_layer=3)
    save_path = tmp_path / "model.pth"
    fabric.save(save_path, {"model": model}, filter={"model": adapter_filter})
    saved = torch.load(save_path)["model"]

    expected = {
        "transformer.h.0.norm_1.bias",
        "transformer.h.0.norm_1.weight",
        "transformer.h.0.norm_2.bias",
        "transformer.h.0.norm_2.weight",
        "transformer.h.1.norm_1.bias",
        "transformer.h.1.norm_1.weight",
        "transformer.h.1.norm_2.bias",
        "transformer.h.1.norm_2.weight",
        "transformer.h.2.attn.adapter_wte.weight",
        "transformer.h.2.attn.gating_factor",
        "transformer.h.2.norm_1.bias",
        "transformer.h.2.norm_1.weight",
        "transformer.h.2.norm_2.bias",
        "transformer.h.2.norm_2.weight",
        "transformer.ln_f.bias",
        "transformer.ln_f.weight",
    }
    assert set(saved) == expected
