from unittest import mock
from pathlib import Path
from urllib.request import urlretrieve

import pytest
import torch

wd = Path(__file__).parent.parent.absolute()


def test_convert_lit_checkpoint(tmp_path):
    from scripts.convert_lit_checkpoint import convert_lit_checkpoint

    ckpt_name = "lit_model_finetuned"

    with pytest.raises(RuntimeError, match="open file failed because of errno 2 on fopen"):
        convert_lit_checkpoint(checkpoint_name=ckpt_name, checkpoint_dir=tmp_path, model_name="falcon-7b")

    ckpt_path = tmp_path / "lit_model_finetuned"
    ckpt_path.touch()
    with mock.patch("scripts.convert_lit_checkpoint.lazy_load") as load:
        convert_lit_checkpoint(checkpoint_name=ckpt_name, checkpoint_dir=tmp_path, model_name="falcon-7b")
    load.assert_called_with(ckpt_path)

    assert {p.name for p in tmp_path.glob("*")} == {ckpt_name, "lit_model_finetuned.bin"}


@torch.inference_mode()
def test_against_original_falcon_40b():
    file_path = wd / "tests" / "original_falcon_40b.py"
    url = "https://gist.githubusercontent.com/carmocca/feed39b1bc65a29f73c1cecc58a01167/raw/a9a65f2b93716b3c09ec9f354d535ae5953de08f/original_falcon_40b.py"
    if not file_path.is_file():
        urlretrieve(url=url, filename=file_path)

    from tests.original_falcon_40b import RWConfig, RWForCausalLM
    from lit_gpt import Config, GPT
    from scripts.convert_lit_checkpoint import copy_weights_falcon as copy_to_theirs
    from scripts.convert_hf_checkpoint import copy_weights_falcon as copy_to_ours

    theirs_config = RWConfig(
        hidden_size=32, n_head=8, n_head_kv=4, n_layer=2, parallel_attn=True, vocab_size=65024, bias=False
    )

    theirs_model = RWForCausalLM(theirs_config)
    theirs_state_dict = theirs_model.state_dict()

    ours_config = Config.from_name("falcon-40b", n_layer=2, n_head=8, n_query_groups=4, n_embd=32)
    ours_model = GPT(ours_config)
    ours_state_dict = {}
    copy_to_ours("40b", ours_state_dict, theirs_state_dict)
    ours_model.load_state_dict(ours_state_dict)

    converted_model = RWForCausalLM(theirs_config)
    converted_state_dict = {}
    copy_to_theirs("40b", converted_state_dict, ours_state_dict)
    converted_model.load_state_dict(converted_state_dict)

    # test converted model matches their model
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    converted_y = converted_model(x)["logits"]
    their_y = theirs_model(x)["logits"]
    torch.testing.assert_close(converted_y, their_y)

    # test our model matches their model
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    ours_y = ours_model(x)
    their_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, their_y)
