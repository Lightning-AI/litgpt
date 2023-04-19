import torch
from dataclasses import asdict
import pytest
import sys


@pytest.mark.skipif(sys.platform == "win32", reason="EmptyInitOnDevice on CPU not working for Windows.")
@pytest.mark.parametrize("model_size", ["7B", "13B", "30B", "65B"])
def test_config_identical(model_size, lit_llama):
    import lit_llama.adapter as llama_adapter
    import lit_llama.model as llama
    from lit_llama.utils import EmptyInitOnDevice

    llama_config = asdict(llama.LLaMAConfig.from_name(model_size))
    adapter_config = asdict(llama_adapter.LLaMAConfig.from_name(model_size))

    del adapter_config["adapter_prompt_length"]
    del adapter_config["adapter_start_layer"]
    assert adapter_config == llama_config

    with EmptyInitOnDevice():
        llama_model = llama.LLaMA.from_name(model_size)
        adapter_model = llama_adapter.LLaMA.from_name(model_size)
        assert llama_model.lm_head.weight.shape == adapter_model.lm_head.weight.shape
