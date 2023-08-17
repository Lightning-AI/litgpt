from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock

import torch
from lightning import Fabric


def test_config_identical():
    import lit_gpt.adapter_v2 as gpt_adapter
    import lit_gpt.model as gpt

    name = "pythia-70m"
    with Fabric(accelerator="cpu").init_module(empty_init=True):
        base_model = gpt.GPT.from_name(name)
        adapter_model = gpt_adapter.GPT.from_name(name)

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
        "lm_head.adapter_bias",
        "lm_head.adapter_scale",
        "transformer.ln_f.bias",
        "transformer.ln_f.weight",
        "transformer.h.2.attn.adapter_wte.weight",
        "transformer.h.2.attn.gating_factor",
    }
    for layer in range(3):
        for param in (
            "attn.attn.adapter_bias",
            "attn.attn.adapter_scale",
            "attn.proj.adapter_bias",
            "attn.proj.adapter_scale",
            "mlp.fc.adapter_bias",
            "mlp.fc.adapter_scale",
            "mlp.proj.adapter_bias",
            "mlp.proj.adapter_scale",
            "norm_1.bias",
            "norm_1.weight",
            "norm_2.bias",
            "norm_2.weight",
        ):
            expected.add(f"transformer.h.{layer}.{param}")
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

    assert {p.name for p in tmp_path.glob("*.pth")} == {
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


def test_adapter_v2_gpt_init_weights():
    from lit_gpt.adapter_v2 import GPT, Config

    config = Config(n_layer=1, n_head=6, n_embd=12, block_size=1, vocab_size=1, adapter_start_layer=0)
    model = GPT(config)

    for param in (model.transformer.h[0].attn.gating_factor, model.lm_head.adapter_bias):
        assert (param == 0).all()
        torch.nn.init.constant_(param, 1.23)
        assert (param != 0).any()
        model.apply(model._init_weights)
        assert (param == 0).all()


def test_base_model_can_be_adapter_v2_loaded():
    from lit_gpt.adapter_v2 import GPT as AdapterV2GPT
    from lit_gpt.adapter_v2 import adapter_filter
    from lit_gpt.model import GPT as BaseGPT

    base_model = BaseGPT.from_name("pythia-70m", bias=True, n_layer=2)
    base_model_state_dict = base_model.state_dict()
    lora_model = AdapterV2GPT.from_name("pythia-70m", bias=True, n_layer=2, adapter_start_layer=0)
    keys = lora_model.load_state_dict(base_model_state_dict, strict=False)
    assert not keys.unexpected_keys
    for k in keys.missing_keys:
        assert adapter_filter(k, None)
