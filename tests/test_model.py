import sys
from pathlib import Path
from urllib.request import urlretrieve

import pytest
import torch
from transformers import GPTNeoXForCausalLM, PretrainedConfig

wd = Path(__file__).parent.parent.absolute()


@torch.inference_mode()
@pytest.mark.parametrize("rotary_pct", (0.25, 1))
@pytest.mark.parametrize("batch_size", (1, 3))
@pytest.mark.parametrize("n_embd", (16, 32))
@pytest.mark.parametrize("parallel_residual", (False, True))
@pytest.mark.parametrize("kv_cache", (False, True))
def test_against_hf_model(rotary_pct, batch_size, n_embd, parallel_residual, kv_cache) -> None:
    import lit_parrot
    from scripts.convert_hf_checkpoint import copy_weights_gpt_neox

    block_size = 64
    # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json#L24
    vocab_size = 100
    n_layer = 4
    n_head = 8
    batch_size = 3

    ours_config = lit_parrot.Config(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        rotary_percentage=rotary_pct,
        parallel_residual=parallel_residual,
    )
    assert ours_config.padded_vocab_size == 512
    theirs_config = PretrainedConfig(
        hidden_act="gelu",
        hidden_size=n_embd,
        num_attention_heads=n_head,
        num_hidden_layers=n_layer,
        initializer_range=0.02,
        intermediate_size=n_embd * 4,
        layer_norm_eps=1e-05,
        max_position_embeddings=block_size,
        rotary_emb_base=10000,
        rotary_pct=rotary_pct,
        vocab_size=ours_config.padded_vocab_size,
        use_parallel_residual=parallel_residual,
        use_cache=kv_cache,
    )

    state_dict = {}
    theirs_model = GPTNeoXForCausalLM(theirs_config)
    # load the hf initialization into our model
    copy_weights_gpt_neox(state_dict, theirs_model.state_dict())
    ours_model = lit_parrot.Parrot(ours_config)
    ours_model.load_state_dict(state_dict)

    token_sample = torch.randint(0, ours_config.padded_vocab_size, size=(batch_size, block_size), dtype=torch.int64)

    theirs_embed = theirs_model.gpt_neox.embed_in(token_sample)
    ours_embed = ours_model.transformer.wte(token_sample)
    torch.testing.assert_close(ours_embed, theirs_embed)

    rope = ours_model.build_rope_cache(token_sample)
    mask = ours_model.build_mask_cache(token_sample)
    if kv_cache:
        (theirs_block_out, theirs_kv_cache) = theirs_model.gpt_neox.layers[0](theirs_embed, use_cache=True)
        head_size = n_embd // n_head
        k_cache_shape = (batch_size, n_head, block_size, rope[0].size(-1) + head_size - int(rotary_pct * head_size))
        v_cache_shape = (batch_size, n_head, block_size, head_size)
        ours_kv_cache = torch.zeros(k_cache_shape), torch.zeros(v_cache_shape)
        (ours_block_out, ours_kv_cache) = ours_model.transformer.h[0](
            ours_embed, rope, mask, block_size, torch.arange(block_size), ours_kv_cache
        )
        for ours_cache, theirs_cache in zip(ours_kv_cache, theirs_kv_cache):
            torch.testing.assert_close(ours_cache, theirs_cache)
    else:
        (theirs_block_out,) = theirs_model.gpt_neox.layers[0](theirs_embed)
        ours_block_out, _ = ours_model.transformer.h[0](ours_embed, rope, mask, block_size)
    torch.testing.assert_close(ours_block_out, theirs_block_out)

    theirs = theirs_model(token_sample)["logits"]
    ours = ours_model(token_sample)
    torch.testing.assert_close(ours, theirs)


@torch.inference_mode()
def test_against_original_falcon_40b():
    file_path = wd / "tests" / "original_falcon_40b.py"
    url = "https://gist.githubusercontent.com/carmocca/feed39b1bc65a29f73c1cecc58a01167/raw/a9a65f2b93716b3c09ec9f354d535ae5953de08f/original_falcon_40b.py"
    if not file_path.is_file():
        urlretrieve(url=url, filename=file_path)

    from tests.original_falcon_40b import RWConfig, RWForCausalLM
    from lit_parrot import Config, Parrot
    from scripts.convert_hf_checkpoint import copy_weights_falcon

    ours_config = Config.from_name("falcon-40b", n_layer=2, n_head=8, n_query_groups=4, n_embd=32)
    theirs_config = RWConfig(
        hidden_size=32, n_head=8, n_head_kv=4, n_layer=2, parallel_attn=True, vocab_size=65024, bias=False
    )

    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)

    theirs_model = RWForCausalLM(theirs_config)
    y_theirs = theirs_model(x)["logits"]

    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_falcon("40b", state_dict, theirs_state_dict)

    ours_model = Parrot(ours_config)
    ours_model.load_state_dict(state_dict)
    y_ours = ours_model(x)

    torch.testing.assert_close(y_ours, y_theirs)


@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="torch.compile not supported on this platform")
@torch.inference_mode()
def test_model_compile():
    import lit_parrot

    config = lit_parrot.Config(block_size=8, vocab_size=8, n_layer=2, n_head=2, n_embd=4)
    model = lit_parrot.Parrot(config)
    model.apply(model._init_weights)

    model = torch.compile(model)

    sample = torch.randint(model.config.vocab_size, size=(2, model.config.block_size), dtype=torch.int64)
    for _ in range(3):
        _ = model(sample)
