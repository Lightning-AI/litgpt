# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest


def test_lightningmodule_state_dict():
    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    from pretrain.openwebtext_trainer import LightningGPTModule

    config = Config.from_name("pythia-14m")
    model = GPT(config)
    lm = LightningGPTModule(config)

    # forgot configure_model
    with pytest.raises(RuntimeError, match="forgot"):
        lm.state_dict()
    with pytest.raises(RuntimeError, match="forgot"):
        lm.load_state_dict({})

    lm.configure_model()

    lm_state_dict = lm.state_dict()
    # the state dict is the same so that the lightningmodule's checkpoints do not need to be converted
    assert set(model.state_dict()) == set(lm_state_dict)
    # the state dict can be loaded back
    lm.load_state_dict(lm_state_dict, strict=True)
