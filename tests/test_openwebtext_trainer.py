import torch


def test_lightningmodule_state_dict(tmp_path):
    from lit_gpt.config import Config
    from lit_gpt.model import GPT
    from pretrain.openwebtext_trainer import LightningGPTModule

    config = Config.from_name("pythia-14m")
    model = GPT(config)
    lm = LightningGPTModule(config)
    lm.configure_model()

    lm_state_dict = lm.state_dict()
    ckpt_path = tmp_path / "foo.ckpt"
    torch.save(lm_state_dict, ckpt_path)

    # the state dict is the same so that the lightningmodule's checkpoints do not need to be converted
    assert set(model.state_dict()) == set(lm_state_dict)

    # the state dict can be loaded back
    lm.load_state_dict(lm_state_dict, strict=True)
