import os
import pathlib
import sys
import tempfile

import pytest
import torch


class ATensor(torch.Tensor):
    pass


def test_lazy_load_basic():
    import lit_parrot.utils

    with tempfile.TemporaryDirectory() as tmpdirname:
        m = torch.nn.Linear(5, 3)
        path = pathlib.Path(tmpdirname)
        fn = str(path / "test.pt")
        torch.save(m.state_dict(), fn)
        with lit_parrot.utils.lazy_load(fn) as sd_lazy:
            assert "NotYetLoadedTensor" in str(next(iter(sd_lazy.values())))
            m2 = torch.nn.Linear(5, 3)
            m2.load_state_dict(sd_lazy)

        x = torch.randn(2, 5)
        actual = m2(x)
        expected = m(x)
        torch.testing.assert_close(actual, expected)


def test_lazy_load_subclass():
    import lit_parrot.utils

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = pathlib.Path(tmpdirname)
        fn = str(path / "test.pt")
        t = torch.randn(2, 3)[:, 1:]
        sd = {1: t, 2: torch.nn.Parameter(t), 3: torch.Tensor._make_subclass(ATensor, t)}
        torch.save(sd, fn)
        with lit_parrot.utils.lazy_load(fn) as sd_lazy:
            for k in sd.keys():
                actual = sd_lazy[k]
                expected = sd[k]
                torch.testing.assert_close(actual._load_tensor(), expected)


def test_find_multiple():
    from lit_parrot.utils import find_multiple

    assert find_multiple(17, 5) == 20
    assert find_multiple(30, 7) == 35
    assert find_multiple(10, 2) == 10
    assert find_multiple(5, 10) == 10
    assert find_multiple(50254, 128) == 50304
    assert find_multiple(50254, 256) == 50432
    assert find_multiple(50254, 512) == 50688


@pytest.mark.skipif(sys.platform == "win32", reason="match fails on windows. why did they have to use backslashes?")
def test_check_valid_checkpoint_dir(tmp_path):
    from lit_parrot.utils import check_valid_checkpoint_dir

    os.chdir(tmp_path)

    match = "must contain the files: 'lit_model.pth', 'lit_config.json', 'tokenizer.json' and 'tokenizer_config.json'."
    with pytest.raises(OSError, match=match):
        check_valid_checkpoint_dir(tmp_path)

    checkpoint_dir = tmp_path / "checkpoints" / "stabilityai" / "stablelm-base-alpha-3b"
    match = (
        r"checkpoint_dir '\/.*checkpoints\/stabilityai\/stablelm-base-alpha-3b' is not.*\nPlease, follow"
        r" the instructions.*download_stablelm\.md\n\nYou can download:\n \* stablelm-base-alpha-3b\n \* stablelm-base-"
        r"alpha-7b\n"
    )
    with pytest.raises(OSError, match=match):
        check_valid_checkpoint_dir(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True)
    match = (
        r"checkpoint_dir '.*foo' is not.*\nPlease, follow the instructions.*download_stablelm\.md\n\nYou"
        r" have downloaded locally:\n --checkpoint_dir '.+checkpoints\/stabilityai\/stablelm-base-alpha-3b'\n\nYou can"
        r" download:\n \* stablelm-base-alpha-7b\n"
    )
    with pytest.raises(OSError, match=match):
        check_valid_checkpoint_dir(tmp_path / "foo")


def test_incremental_write(tmp_path):
    from lit_parrot.utils import incremental_save

    sd = {str(k): torch.randn(5, 10) for k in range(3)}
    sd_expected = {k: v.clone() for k, v in sd.items()}
    fn = str(tmp_path / "test.pt")
    with incremental_save(fn) as f:
        sd["0"] = f.store_early(sd["0"])
        sd["2"] = f.store_early(sd["2"])
        f.save(sd)
    sd_actual = torch.load(fn)
    assert sd_actual.keys() == sd_expected.keys()
    for k, v_expected in sd_expected.items():
        v_actual = sd_actual[k]
        torch.testing.assert_close(v_expected, v_actual)
