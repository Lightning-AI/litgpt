# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from contextlib import redirect_stderr
from io import StringIO

import pytest
import torch
import torch.nn.functional as F
from conftest import RunIf
from lightning import Fabric


def test_find_multiple():
    from lit_gpt.utils import find_multiple

    assert find_multiple(17, 5) == 20
    assert find_multiple(30, 7) == 35
    assert find_multiple(10, 2) == 10
    assert find_multiple(5, 10) == 10
    assert find_multiple(50254, 128) == 50304
    assert find_multiple(50254, 256) == 50432
    assert find_multiple(50254, 512) == 50688


# match fails on windows. why did they have to use backslashes?
@RunIf(skip_windows=True)
def test_check_valid_checkpoint_dir(tmp_path):
    from lit_gpt.utils import check_valid_checkpoint_dir

    os.chdir(tmp_path)

    out = StringIO()
    with pytest.raises(SystemExit), redirect_stderr(out):
        check_valid_checkpoint_dir(tmp_path)
    out = out.getvalue().strip()
    expected = f"""
--checkpoint_dir '{str(tmp_path.absolute())}' is missing the files: ['lit_model.pth', 'lit_config.json', 'tokenizer.json OR tokenizer.model', 'tokenizer_config.json'].
Find download instructions at https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials

See all download options by running:
 python scripts/download.py
    """.strip()
    assert out == expected

    out = StringIO()
    checkpoint_dir = tmp_path / "checkpoints" / "stabilityai" / "stablelm-base-alpha-3b"
    with pytest.raises(SystemExit), redirect_stderr(out):
        check_valid_checkpoint_dir(checkpoint_dir)
    out = out.getvalue().strip()
    expected = f"""
--checkpoint_dir '{str(checkpoint_dir.absolute())}' is not a checkpoint directory.
Find download instructions at https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials

See all download options by running:
 python scripts/download.py
    """.strip()
    assert out == expected

    out = StringIO()
    checkpoint_dir.mkdir(parents=True)
    foo_checkpoint_dir = tmp_path / "foo"
    with pytest.raises(SystemExit), redirect_stderr(out):
        check_valid_checkpoint_dir(foo_checkpoint_dir)
    out = out.getvalue().strip()
    expected = f"""
--checkpoint_dir '{str(foo_checkpoint_dir.absolute())}' is not a checkpoint directory.
Find download instructions at https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials

You have downloaded locally:
 --checkpoint_dir '{str(checkpoint_dir.absolute())}'

See all download options by running:
 python scripts/download.py
    """.strip()
    assert out == expected


def test_incremental_write(tmp_path):
    from lit_gpt.utils import incremental_save

    sd = {str(k): torch.randn(5, 10) for k in range(3)}
    sd["0"].someattr = 1
    sd_expected = {k: v.clone() for k, v in sd.items()}
    fn = str(tmp_path / "test.pt")
    with incremental_save(fn) as f:
        sd["0"] = f.store_early(sd["0"])
        sd["2"] = f.store_early(sd["2"])
        f.save(sd)
    sd_actual = torch.load(fn)
    assert sd_actual.keys() == sd_expected.keys()
    assert sd_actual["0"].someattr == 1  # requires PyTorch 2.0+
    for k, v_expected in sd_expected.items():
        v_actual = sd_actual[k]
        torch.testing.assert_close(v_expected, v_actual)


@pytest.mark.parametrize("B", (1, 2))
@pytest.mark.parametrize("with_ignore_index", (True, False))
def test_chunked_cross_entropy(with_ignore_index, B):
    from lit_gpt.utils import chunked_cross_entropy

    V = 50
    T = 25
    regular_logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))

    if with_ignore_index:
        targets[:, [1, 4, 10, 19]] = -1

    baseline_loss = F.cross_entropy(
        regular_logits.reshape(-1, regular_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=(-1 if with_ignore_index else -100),
    )
    regular_loss = chunked_cross_entropy(regular_logits, targets, chunk_size=0)
    assert torch.equal(baseline_loss, regular_loss)
    assert regular_loss.numel() == 1

    chunked_loss = chunked_cross_entropy(regular_logits, targets, chunk_size=10)
    torch.testing.assert_close(chunked_loss, regular_loss)
    torch.testing.assert_close(chunked_loss, baseline_loss)

    logit_chunk_size = 6
    assert T % logit_chunk_size != 0  # ensure leftover
    chunked_logits = list(regular_logits.split(logit_chunk_size, dim=1))
    chunked_loss = chunked_cross_entropy(chunked_logits, targets, chunk_size=0)
    torch.testing.assert_close(chunked_loss, regular_loss)
    torch.testing.assert_close(chunked_loss, baseline_loss)

    chunked_loss = chunked_cross_entropy(chunked_logits, targets, chunk_size=10)
    torch.testing.assert_close(chunked_loss, regular_loss)
    torch.testing.assert_close(chunked_loss, baseline_loss)


def test_num_parameters():
    from lit_gpt.utils import num_parameters

    model = torch.nn.Linear(2, 2)
    assert num_parameters(model) == 6
    assert num_parameters(model, requires_grad=True) == 6
    assert num_parameters(model, requires_grad=False) == 0

    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad = False
    assert num_parameters(model) == 6
    assert num_parameters(model, requires_grad=True) == 4
    assert num_parameters(model, requires_grad=False) == 2


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("mode", ["nf4", "nf4-dq", "fp4", "fp4-dq", "int8", "int8-training"])
@pytest.mark.skip("To be fixed")
def test_num_parameters_bitsandbytes(mode):
    from lightning.fabric.plugins import BitsandbytesPrecision

    from lit_gpt import GPT
    from lit_gpt.utils import num_parameters

    plugin = BitsandbytesPrecision(mode=mode)
    fabric = Fabric(plugins=plugin, accelerator="cuda", devices=1)

    model = torch.nn.Linear(10, 10)
    model = fabric.setup(model)
    assert num_parameters(model) == 110

    with fabric.init_module(empty_init=True):
        model = GPT.from_name("pythia-14m")
    assert num_parameters(model) == 14067712


def test_cycle_iterator():
    from lit_gpt.utils import CycleIterator

    iterator = CycleIterator([])
    with pytest.raises(StopIteration):
        next(iterator)

    iterator = CycleIterator(range(3))
    assert iterator.epoch == 0
    assert next(iterator) == 0
    assert iterator.epoch == 0
    assert next(iterator) == 1
    assert iterator.epoch == 0
    assert next(iterator) == 2
    assert iterator.epoch == 0
    assert next(iterator) == 0
    assert iterator.epoch == 1
