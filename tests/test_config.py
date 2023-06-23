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
