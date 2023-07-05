from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock

import torch
from lightning import Fabric


def test_config_identical():
    import lit_gpt.adapter as gpt_adapter
    from lit_gpt.adapter_v2 import adapter_v2_linear_with_bias_and_scale
    import lit_gpt.model as gpt

    name = "pythia-70m"
    with Fabric(accelerator="cpu").init_module(empty_init=True):
        base_model = gpt.GPT.from_name(name)
        adapter_model = gpt_adapter.GPT.from_name(name)

        for module in adapter_model.modules():
            if isinstance(module, torch.nn.Linear):
                adapter_v2_linear_with_bias_and_scale(module)

    assert not hasattr(base_model.transformer.h[2].attn.attn, "adapter_bias")
    assert not hasattr(base_model.transformer.h[2].attn.attn, "adapter_scale")
    assert hasattr(adapter_model.transformer.h[2].attn.attn, "adapter_bias")
    assert hasattr(adapter_model.transformer.h[2].attn.attn, "adapter_scale")


def test_adapter_v2_filter(tmp_path):
    from lit_gpt.adapter_v2 import GPT, adapter_filter

    fabric = Fabric(devices=1)
    model = GPT.from_name("pythia-70m", n_layer=3)
    save_path = tmp_path / "model.pth"
    fabric.save(save_path, {"model": model}, filter={"model": adapter_filter})
    saved = torch.load(save_path)["model"]

    expected = {
        "transformer.h.0.norm_1.bias",
        "transformer.h.0.norm_1.weight",
        "transformer.h.0.norm_2.bias",
        "transformer.h.0.norm_2.weight",
        "transformer.h.1.norm_1.bias",
        "transformer.h.1.norm_1.weight",
        "transformer.h.1.norm_2.bias",
        "transformer.h.1.norm_2.weight",
        "transformer.h.2.attn.adapter_wte.weight",
        "transformer.h.2.attn.gating_factor",
        "transformer.h.2.norm_1.bias",
        "transformer.h.2.norm_1.weight",
        "transformer.h.2.norm_2.bias",
        "transformer.h.2.norm_2.weight",
        "transformer.ln_f.bias",
        "transformer.ln_f.weight",
    }
    assert set(saved) == expected


def test_adapter_v2_script(tmp_path, fake_checkpoint_dir, monkeypatch):
    import finetune.adapter_v2 as module

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

    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8, adapter_start_layer=0)
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

    assert set(p.name for p in tmp_path.glob("*.pth")) == {
        "iter-000001-ckpt.pth",
        "iter-000003-ckpt.pth",
        "iter-000005-ckpt.pth",
        "lit_model_adapter_finetuned.pth",
    }
    assert (tmp_path / "version_0" / "metrics.csv").is_file()

    logs = stdout.getvalue()
    assert logs.count("optimizer.step") == module.max_iters
    assert logs.count("val loss") == module.max_iters // module.eval_interval
    assert "of trainable parameters: 552" in logs
