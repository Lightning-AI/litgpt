import sys
from dataclasses import asdict

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="EmptyInitOnDevice on CPU not working for Windows.")
@pytest.mark.parametrize("name", ["pythia-70m", "stablelm-base-alpha-3b"])
def test_config_identical(name, lit_stablelm):
    import lit_stablelm.adapter as stablelm_adapter
    import lit_stablelm.model as stablelm
    from lit_stablelm.utils import EmptyInitOnDevice

    base_config = asdict(stablelm.Config.from_name(name))
    adapter_config = asdict(stablelm_adapter.Config.from_name(name))

    del adapter_config["adapter_prompt_length"]
    del adapter_config["adapter_start_layer"]
    assert adapter_config == base_config

    with EmptyInitOnDevice():
        base_model = stablelm.StableLM.from_name(name)
        adapter_model = stablelm_adapter.StableLM.from_name(name)
        assert adapter_model.lm_head.weight.shape == base_model.lm_head.weight.shape
