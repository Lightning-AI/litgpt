import tempfile
import pathlib

import torch


class ATensor(torch.Tensor):
    pass


def test_lazy_load_basic(lit_llama):
    import lit_llama.utils

    with tempfile.TemporaryDirectory() as tmpdirname:
        m = torch.nn.Linear(5, 3)
        path = pathlib.Path(tmpdirname)
        fn = str(path / "test.pt")
        torch.save(m.state_dict(), fn)
        with lit_llama.utils.lazy_load(fn) as sd_lazy:
            assert "NotYetLoadedTensor" in str(next(iter(sd_lazy.values())))
            m2 = torch.nn.Linear(5, 3)
            m2.load_state_dict(sd_lazy)

        x = torch.randn(2, 5)
        actual = m2(x)
        expected = m(x)
        torch.testing.assert_close(actual, expected)


def test_lazy_load_subclass(lit_llama):
    import lit_llama.utils

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = pathlib.Path(tmpdirname)
        fn = str(path / "test.pt")
        t = torch.randn(2, 3)[:, 1:]
        sd = {
            1: t,
            2: torch.nn.Parameter(t),
            3: torch.Tensor._make_subclass(ATensor, t),
        }
        torch.save(sd, fn)
        with lit_llama.utils.lazy_load(fn) as sd_lazy:
            for k in sd.keys():
                actual = sd_lazy[k]
                expected = sd[k]
                torch.testing.assert_close(actual._load_tensor(), expected)
