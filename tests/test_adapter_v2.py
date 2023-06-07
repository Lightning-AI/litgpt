import sys

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="EmptyInitOnDevice on CPU not working for Windows.")
@pytest.mark.parametrize("name", ["pythia-70m", "stablelm-base-alpha-3b"])
def test_config_identical(name):
    import torch.nn as nn
    import lit_parrot.adapter as parrot_adapter
    from lit_parrot.adapter_v2 import adapter_v2_linear_with_bias_and_scale
    import lit_parrot.model as parrot
    from lit_parrot.utils import EmptyInitOnDevice

    with EmptyInitOnDevice():
        base_model = parrot.Parrot.from_name(name)
        adapter_model = parrot_adapter.Parrot.from_name(name)

        for module in adapter_model.modules():
            if isinstance(module, nn.Linear):
                adapter_v2_linear_with_bias_and_scale(module)

        print(adapter_model.transformer.h[2].attn.attn.adapter_bias)
        assert not hasattr(base_model.transformer.h[2].attn.attn, "adapter_bias")
        assert not hasattr(base_model.transformer.h[2].attn.attn, "adapter_scale")
        assert hasattr(adapter_model.transformer.h[2].attn.attn, "adapter_bias")
        assert hasattr(adapter_model.transformer.h[2].attn.attn, "adapter_scale")


def test_adapter_state_only():
    from lit_parrot.adapter_v2 import adapter_state_only, Parrot

    model = Parrot.from_name("pythia-70m", n_layer=3)

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
    with adapter_state_only(model):
        assert set(model.state_dict()) == expected
    assert len(model.state_dict()) > len(expected)
