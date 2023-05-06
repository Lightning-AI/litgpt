import functools
from pathlib import Path

import torch
import pytest
import sys

from transformers import GPTNeoXForCausalLM, PretrainedConfig


wd = Path(__file__).parent.parent.absolute()


@functools.lru_cache(maxsize=1)
def load_convert_script():
    sys.path.append(str(wd / "scripts"))

    import convert_hf_checkpoint

    return convert_hf_checkpoint


@torch.inference_mode()
@pytest.mark.parametrize("rotary_pct", (0.25, 1))
@pytest.mark.parametrize("batch_size", (1, 3))
@pytest.mark.parametrize("n_embd", (16, 32))
@pytest.mark.parametrize("parallel_residual", (False, True))
def test_against_hf_model(rotary_pct, batch_size, n_embd, parallel_residual, lit_stablelm) -> None:
    block_size = 64
    # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json#L24
    vocab_size = 100
    n_layer = 4
    n_head = 8
    batch_size = 3

    ours_config = lit_stablelm.StableLMConfig(
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
        use_cache=False,
    )

    ours_model = lit_stablelm.StableLM(ours_config)
    state_dict = ours_model.state_dict()
    theirs_model = GPTNeoXForCausalLM(theirs_config)

    convert_hf_checkpoint = load_convert_script()
    # load the hf initialization into our model
    convert_hf_checkpoint.copy_weights(state_dict, theirs_model.state_dict())
    ours_model.load_state_dict(state_dict)

    token_sample = torch.randint(0, ours_config.padded_vocab_size, size=(batch_size, block_size), dtype=torch.int64)

    theirs_embed = theirs_model.gpt_neox.embed_in(token_sample)
    ours_embed = ours_model.transformer.wte(token_sample)
    torch.testing.assert_close(ours_embed, theirs_embed)

    (theirs_block_out,) = theirs_model.gpt_neox.layers[0](theirs_embed)
    ours_block_out = ours_model.transformer.h[0](ours_embed)
    torch.testing.assert_close(ours_block_out, theirs_block_out)

    theirs = theirs_model(token_sample)["logits"]
    ours = ours_model(token_sample)
    torch.testing.assert_close(ours, theirs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@torch.inference_mode()
def test_model_bfloat16(lit_stablelm) -> None:
    from lit_stablelm.utils import EmptyInitOnDevice

    block_size = 64
    vocab_size = 32000
    n_layer = 16
    n_head = 16
    n_embd = 32

    config = lit_stablelm.StableLMConfig(
        block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd
    )
    model = lit_stablelm.StableLM(config)
    model.apply(model._init_weights)

    batch_size = 3
    token_sample = torch.randint(0, vocab_size, size=(batch_size, block_size), dtype=torch.int64)

    expected = model(token_sample)

    with EmptyInitOnDevice(device="cuda", dtype=torch.bfloat16):
        model2 = lit_stablelm.StableLM(config)
    model2.load_state_dict(model.state_dict(keep_vars=True))

    out = model2(token_sample.cuda()).float().cpu()
    torch.testing.assert_close(out, expected, atol=5e-3, rtol=1e-3)


@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="torch.compile not supported on this platform")
@torch.inference_mode()
def test_model_compile(lit_stablelm):
    config = lit_stablelm.StableLMConfig(block_size=8, vocab_size=8, n_layer=2, n_head=2, n_embd=4)
    model = lit_stablelm.StableLM(config)
    model.apply(model._init_weights)

    model = torch.compile(model)

    sample = torch.randint(model.config.vocab_size, size=(2, model.config.block_size), dtype=torch.int64)
    for _ in range(3):
        _ = model(sample)
