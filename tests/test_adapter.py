from dataclasses import asdict

import pytest
from lightning import Fabric


@pytest.mark.parametrize("name", ["pythia-70m", "stablelm-base-alpha-3b"])
def test_config_identical(name):
    import lit_parrot.adapter as parrot_adapter
    import lit_parrot.model as parrot

    fabric = Fabric(accelerator="cpu")

    base_config = asdict(parrot.Config.from_name(name))
    adapter_config = asdict(parrot_adapter.Config.from_name(name))
    del adapter_config["adapter_prompt_length"]
    del adapter_config["adapter_start_layer"]
    assert adapter_config == base_config

    with fabric.init_module(empty_init=True):
        base_model = parrot.Parrot.from_name(name)
        adapter_model = parrot_adapter.Parrot.from_name(name)
    assert adapter_model.lm_head.weight.shape == base_model.lm_head.weight.shape
