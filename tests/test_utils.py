import os
import pathlib
import sys
import tempfile
from contextlib import redirect_stderr
from io import StringIO

import pytest
import torch
import torch.nn.functional as F


class ATensor(torch.Tensor):
    pass


def test_lazy_load_basic():
    import lit_gpt.utils

    with tempfile.TemporaryDirectory() as tmpdirname:
        m = torch.nn.Linear(5, 3)
        path = pathlib.Path(tmpdirname)
        fn = str(path / "test.pt")
        torch.save(m.state_dict(), fn)
        with lit_gpt.utils.lazy_load(fn) as sd_lazy:
            assert "NotYetLoadedTensor" in str(next(iter(sd_lazy.values())))
            m2 = torch.nn.Linear(5, 3)
            m2.load_state_dict(sd_lazy)

        x = torch.randn(2, 5)
        actual = m2(x)
        expected = m(x)
        torch.testing.assert_close(actual, expected)


def test_lazy_load_subclass():
    import lit_gpt.utils

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = pathlib.Path(tmpdirname)
        fn = str(path / "test.pt")
        t = torch.randn(2, 3)[:, 1:]
        sd = {1: t, 2: torch.nn.Parameter(t), 3: torch.Tensor._make_subclass(ATensor, t)}
        torch.save(sd, fn)
        with lit_gpt.utils.lazy_load(fn) as sd_lazy:
            for k in sd:
                actual = sd_lazy[k]
                expected = sd[k]
                torch.testing.assert_close(actual._load_tensor(), expected)


def test_find_multiple():
    from lit_gpt.utils import find_multiple

    assert find_multiple(17, 5) == 20
    assert find_multiple(30, 7) == 35
    assert find_multiple(10, 2) == 10
    assert find_multiple(5, 10) == 10
    assert find_multiple(50254, 128) == 50304
    assert find_multiple(50254, 256) == 50432
    assert find_multiple(50254, 512) == 50688


@pytest.mark.skipif(sys.platform == "win32", reason="match fails on windows. why did they have to use backslashes?")
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
def test_chunked_cross_entropy(B):
    from lit_gpt.utils import chunked_cross_entropy

    V = 50
    T = 25
    regular_logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))

    baseline_loss = F.cross_entropy(regular_logits.reshape(-1, regular_logits.size(-1)), targets.reshape(-1))
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
