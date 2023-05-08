import torch
from dataclasses import asdict
import pytest
import sys


@pytest.mark.skipif(sys.platform == "win32", reason="EmptyInitOnDevice on CPU not working for Windows.")
@pytest.mark.parametrize("model_size", ["3B", "7B"])
def test_config_identical(model_size, lit_stablelm):
    import lit_stablelm.adapter as stablelm_adapter
    import lit_stablelm.model as stablelm
    from lit_stablelm.utils import EmptyInitOnDevice
    from lit_stablelm.config import Config

    stablelm_config = asdict(Config.from_name(model_size))
    adapter_config = asdict(llama_adapter.StableLMConfig.from_name(model_size))

    del adapter_config["adapter_prompt_length"]
    del adapter_config["adapter_start_layer"]
    assert adapter_config == llama_config

    with EmptyInitOnDevice():
        stablelm_model = stablelm.StableLM.from_name(model_size)
        adapter_model = stablelm_adapter.StableLM.from_name(model_size)
        assert stablelm_model.lm_head.weight.shape == adapter_model.lm_head.weight.shape
