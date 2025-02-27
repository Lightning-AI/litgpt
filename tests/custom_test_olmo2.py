# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from copy import deepcopy
from functools import partial
from unittest import mock

import pytest
import torch
from lightning import Fabric
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.fabric.utilities.init import _materialize_meta_tensors
from torch._dynamo.backends import debugging
from torch.backends.cuda import (
    SDPAParams,
    SDPBackend,
    can_use_efficient_attention,
    can_use_flash_attention,
    flash_sdp_enabled,
    math_sdp_enabled,
    mem_efficient_sdp_enabled,
)
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.falcon import FalconConfig, FalconForCausalLM
from transformers.models.gemma import GemmaConfig, GemmaForCausalLM
from transformers.models.gemma2 import Gemma2Config, Gemma2ForCausalLM
from transformers.models.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral import MistralConfig, MistralForCausalLM
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM
from transformers.models.olmo import OlmoConfig, OlmoForCausalLM
from transformers.models.olmo2 import Olmo2Config, Olmo2ForCausalLM
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

import litgpt.config as config_module
from litgpt import GPT, Config
from litgpt.model import CausalSelfAttention, batched_index_copy_
from litgpt.scripts.convert_hf_checkpoint import (
    copy_weights_falcon,
    copy_weights_gemma_2,
    copy_weights_gpt_neox,
    copy_weights_hf_llama,
    copy_weights_olmo2,
    copy_weights_phi,
    copy_weights_qwen_2_5,
)
from litgpt.scripts.convert_lit_checkpoint import qkv_reassemble as make_qkv_interleaved
from litgpt.utils import _RunIf

@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("OLMo-2-1124-7B", "OLMo-2-1124-13B"))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                _RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_olmo2(model_name, device, dtype):
    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        model_name,
        padded_vocab_size=10000,
        n_layer=1,
        n_head=8,
        n_embd=32,
        n_query_groups=2,
        intermediate_size=86,
    )
    T = 5
    theirs_config = Olmo2Config(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        intermediate_size=ours_config.intermediate_size,
        num_hidden_layers=ours_config.n_layer,
        num_attention_heads=ours_config.n_head,
        num_key_value_heads=ours_config.n_query_groups,
        max_positional_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        attention_bias=ours_config.bias,
        rope_theta=ours_config.rope_base,
    )

    print(ours_config)
    print(theirs_config)
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = Olmo2ForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    print(theirs_state_dict.keys())
    print(len(theirs_state_dict.keys()))
    copy_weights_olmo2(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    print(ours_model.state_dict().keys())
    print(len(ours_model.state_dict().keys()))

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float

    print(ours_y)
    print(theirs_y)
    torch.testing.assert_close(ours_y, theirs_y)

