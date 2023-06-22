import torch

from lightning import Fabric


def test_lora_layer_replacement():
    from lit_parrot.lora import CausalSelfAttention as LoRACausalSelfAttention, Parrot, Config

    config = Config(n_layer=2, n_head=4, n_embd=8, block_size=8, vocab_size=8, r=8, alpha=8, dropout=0.1)
    model = Parrot(config)

    assert isinstance(model.transformer.h[0].attn, LoRACausalSelfAttention)
    assert isinstance(model.transformer.h[1].attn, LoRACausalSelfAttention)


def test_lora_merge_unmerge():
    from lit_parrot.lora import mark_only_lora_as_trainable, Parrot, Config

    config = Config(n_layer=1, n_head=2, n_embd=8, block_size=8, vocab_size=8, r=8, alpha=8, dropout=0.1)
    model = Parrot(config)

    initial_weight = model.transformer.h[0].attn.attn.weight.clone()
    model.train()
    assert torch.equal(model.transformer.h[0].attn.attn.weight, initial_weight)

    # perform an update to the LoRA weights
    mark_only_lora_as_trainable(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    y = model(torch.randint(0, 8, size=(2, 4), dtype=torch.int64))
    torch.cat(y, dim=1).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    # the weight remains unchanged (only lora A and B change)
    assert torch.equal(model.transformer.h[0].attn.attn.weight, initial_weight)

    # 'merge' and then 'unmerge' should neutralize themselves
    weight_before = model.transformer.h[0].attn.attn.weight.clone()
    model.eval()
    assert not torch.equal(model.transformer.h[0].attn.attn.weight, weight_before)
    model.train()
    # note: numerically, `W + (A * B) - (A * B) == W` does not hold exactly
    torch.testing.assert_close(model.transformer.h[0].attn.attn.weight, weight_before)

    # calling eval/train multiple times in a row should not merge/unmerge multiple times
    model.eval()
    assert model.transformer.h[0].attn.attn.merged
    weight_after = model.transformer.h[0].attn.attn.weight.clone()
    model.eval()
    model.eval()
    assert torch.equal(model.transformer.h[0].attn.attn.weight, weight_after)
    model.train()
    assert not model.transformer.h[0].attn.attn.merged
    weight_after = model.transformer.h[0].attn.attn.weight.clone()
    model.train()
    model.train()
    assert torch.equal(model.transformer.h[0].attn.attn.weight, weight_after)


def test_lora_mqa_gqa():
    from lit_parrot.lora import Parrot, Config

    # MHA
    config = Config(n_layer=1, n_head=4, n_embd=8, block_size=1, vocab_size=1, r=2, alpha=8, dropout=0.1)
    assert config.n_query_groups == config.n_head
    model = Parrot(config)
    attn = model.transformer.h[0].attn.attn
    assert attn.weight.shape == (24, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (16, 2)
    assert attn.lora_ind.tolist() == [True] * 8 + [False] * 8 + [True] * 8
    x = torch.randint(0, 8, size=(3, 5, 16), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 24)

    # MQA
    config.n_query_groups = 1
    model = Parrot(config)
    attn = model.transformer.h[0].attn.attn
    assert attn.weight.shape == (12, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (10, 2)
    assert attn.lora_ind.tolist() == [True] * 8 + [False] * 2 + [True] * 2
    x = torch.randint(0, 8, size=(3, 5, 10), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 12)

    # GQA
    config.n_query_groups = 2
    model = Parrot(config)
    attn = model.transformer.h[0].attn.attn
    assert attn.weight.shape == (16, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (12, 2)
    assert attn.lora_ind.tolist() == [True] * 8 + [False] * 4 + [True] * 4
    x = torch.randint(0, 8, size=(3, 5, 12), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 16)


def test_lora_filter(tmp_path):
    from lit_parrot.lora import lora_filter, Parrot

    fabric = Fabric(devices=1)
    model = Parrot.from_name("pythia-70m", n_layer=3, r=1)
    save_path = tmp_path / "model.pth"
    fabric.save(save_path, {"model": model}, filter={"model": lora_filter})
    saved = torch.load(save_path)["model"]

    expected = {
        "transformer.h.1.attn.attn.lora_B",
        "transformer.h.2.attn.attn.lora_B",
        "transformer.h.2.attn.attn.lora_A",
        "transformer.h.1.attn.attn.lora_A",
        "transformer.h.0.attn.attn.lora_A",
        "transformer.h.0.attn.attn.lora_B",
    }
    assert set(saved) == expected
