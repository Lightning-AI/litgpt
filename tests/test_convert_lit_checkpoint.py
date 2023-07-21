from pathlib import Path
from urllib.request import urlretrieve

import pytest
import torch

wd = Path(__file__).parent.parent.absolute()


@torch.inference_mode()
def test_against_original_falcon_40b():
    file_path = wd / "tests" / "original_falcon_40b.py"
    url = "https://gist.githubusercontent.com/carmocca/feed39b1bc65a29f73c1cecc58a01167/raw/a9a65f2b93716b3c09ec9f354d535ae5953de08f/original_falcon_40b.py"
    if not file_path.is_file():
        urlretrieve(url=url, filename=file_path)

    from tests.original_falcon_40b import RWConfig, RWForCausalLM
    from lit_gpt import Config, GPT
    from scripts.convert_lit_checkpoint import copy_weights_falcon

    # the unchanged model config
    our_config = Config.from_name("falcon-40b", n_layer=2, n_head=8, n_query_groups=4, n_embd=32)
    theirs_config = RWConfig(
        hidden_size=32, n_head=8, n_head_kv=4, n_layer=2, parallel_attn=True, vocab_size=65024, bias=False
    )

    ground_truth_their_model = RWForCausalLM(theirs_config)
    ground_truth_state_dict = ground_truth_their_model.state_dict()

    # need a separate state_dict to pass to copy_weights_falcon
    # convert_lit_checkpoint.copy_weights_falcon is expecting lit-gpt format
    our_model = GPT(our_config)
    our_state_dict = our_model.state_dict()
    our_model.load_state_dict(our_state_dict)
    assert "transformer.wte.weight" in our_state_dict.keys()

    their_state_dict = {}
    copy_weights_falcon("40b", their_state_dict, our_state_dict)
    their_model = RWForCausalLM(theirs_config)
    their_model.load_state_dict(their_state_dict)
    assert all(
        [
            torch.equal(their_state_dict[tsdk], our_state_dict[osdk])
            for tsdk, osdk in zip(their_state_dict.keys(), our_state_dict.keys())
        ]
    )
    assert all(litkey in their_state_dict.keys() for litkey in ground_truth_state_dict.keys())
    assert len(their_state_dict.keys()) == len(ground_truth_state_dict.keys())

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    our_y = our_model(x)
    their_y = their_model(x)["logits"]
    torch.testing.assert_close(our_y, their_y)


@torch.inference_mode()
@pytest.mark.skip("not implemented")
def test_against_original_open_llama_3b():
    pass


@torch.inference_mode()
@pytest.mark.skip("not implemented")
@pytest.mark.parametrize("size", ("7b", "70b"))
def test_against_hf_llama2(size):
    pass
