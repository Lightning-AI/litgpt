from dataclasses import asdict

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


def test_adapter_state_only():
    from lit_parrot.adapter import Parrot, adapter_state_only

    model = Parrot.from_name("pythia-70m", n_layer=4)

    expected = {
        "transformer.h.2.attn.adapter_wte.weight",
        "transformer.h.2.attn.gating_factor",
        "transformer.h.3.attn.adapter_wte.weight",
        "transformer.h.3.attn.gating_factor",
    }
    with adapter_state_only(model):
        assert set(model.state_dict()) == expected
    assert len(model.state_dict()) > len(expected)
