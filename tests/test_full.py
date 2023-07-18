from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock

import torch


def test_full_script(tmp_path, fake_checkpoint_dir, monkeypatch):
    import finetune.full as module

    module.gradient_accumulation_iters = 1
    module.save_interval = 2
    module.eval_interval = 2
    module.eval_iters = 2
    module.max_iters = 6

    data = [
        {"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
    ]
    torch.save(data, tmp_path / "train.pt")
    torch.save(data, tmp_path / "test.pt")

    from lit_gpt.config import name_to_config

    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    monkeypatch.setitem(name_to_config, "tmp", model_config)

    load_mock = Mock()
    load_mock.return_value = load_mock
    load_mock.__enter__ = Mock()
    load_mock.__exit__ = Mock()
    monkeypatch.setattr(module, "lazy_load", load_mock)

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **kwargs: torch.tensor([3, 2, 1], **kwargs)
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    stdout = StringIO()
    with redirect_stdout(stdout):
        module.setup(data_dir=tmp_path, checkpoint_dir=fake_checkpoint_dir, out_dir=tmp_path, precision="32-true")

    assert {p.name for p in tmp_path.glob("*.pth")} == {
        "iter-000001-ckpt.pth",
        "iter-000003-ckpt.pth",
        "iter-000005-ckpt.pth",
        "lit_model_finetuned.pth",
    }
    assert (tmp_path / "version_0" / "metrics.csv").is_file()

    logs = stdout.getvalue()
    assert logs.count("optimizer.step") == module.max_iters
    assert logs.count("val loss") == module.max_iters // module.eval_interval
    assert "of trainable parameters: 1,888" in logs
