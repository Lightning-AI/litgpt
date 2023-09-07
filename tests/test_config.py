import json


def test_config():
    from lit_gpt import Config

    config = Config()
    assert config.block_size == 4096

    config = Config(block_size=2048)
    assert config.block_size == 2048

    config = Config.from_name("pythia-70m")
    assert config.block_size == 2048

    config = Config.from_name("pythia-70m", block_size=4096)
    assert config.block_size == 4096


def test_legacy_args(tmp_path):
    from lit_gpt import Config

    config = Config.from_name("pythia-70m", condense_ratio=2)
    assert not hasattr(config, "condense_ratio")
    assert config.rope_condense_ratio == 2

    json_path = tmp_path / "config.json"
    with open(json_path, "w") as fp:
        json.dump({"condense_ratio": 3}, fp)

    config = Config.from_json(json_path)
    assert not hasattr(config, "condense_ratio")
    assert config.rope_condense_ratio == 3
    config = Config.from_json(json_path, condense_ratio=2)
    assert not hasattr(config, "condense_ratio")
    assert config.rope_condense_ratio == 2
