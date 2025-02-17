# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import ANY, Mock, call

import lightning as L
import pytest
import torch
import yaml

import litgpt.generate.speculative_decoding as generate
from litgpt import GPT, Config
from litgpt.generate.speculative_decoding import sample


@pytest.mark.parametrize("max_seq_length", (10, 15, 20, 25))
@pytest.mark.parametrize("speculative_k", (1, 2, 3))
def test_generate(max_seq_length, speculative_k):
    # create a prompt
    T = 5
    input_idx = torch.arange(0, T)
    max_new_tokens = max_seq_length - T

    # prepare models
    draft_model = GPT(Config(vocab_size=16, block_size=64, n_layer=1, n_head=4, n_embd=8))
    target_model = GPT(Config(vocab_size=16, block_size=128, n_layer=2, n_head=8, n_embd=16))
    for model in (draft_model, target_model):
        model.max_seq_length = max_seq_length
        model.set_kv_cache(batch_size=1)

    # generate tokens
    out, acceptance_rate = generate.generate(draft_model, target_model, input_idx, T + max_new_tokens, top_k=1, speculative_k=speculative_k)

    # validate
    assert out.size(0) == T + max_new_tokens - 1, (out.size(0), T + max_new_tokens - 1)
    assert 0.0 <= acceptance_rate <= 1.0
