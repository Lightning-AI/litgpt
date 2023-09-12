from pathlib import Path

import lightning as L
import torch

wd = Path(__file__).parent.parent.absolute()


def test_convert_lora_checkpoint(tmp_path):
    import json
    from dataclasses import asdict

    from lit_gpt.lora import GPT as LoRAGPT
    from lit_gpt.lora import Config as LoRAConfig
    from lit_gpt.model import GPT, Config
    from scripts.convert_lit_lora_checkpoint import convert_lit_lora_checkpoint

    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    model_name = "Llama-2-7b-hf"

    converted_checkpoint_path = tmp_path / "converted_merged_lora_model.bin"
    lora_path = tmp_path / "lit_model_lora.pth"
    lora_config_path = tmp_path / "lit_lora_config.json"
    ours_ckpt_path = tmp_path / "lit_model.pth"

    # llama checkpoint
    ours_config = Config.from_name(
        name=model_name,
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_size=8,
        vocab_size=8,
    )
    ours_model = GPT(ours_config)
    initial_weight = ours_model.transformer.h[0].attn.proj.weight.clone()
    torch.save(ours_model.state_dict(), ours_ckpt_path)

    # LoRA finetuned checkpoint
    lora_config = LoRAConfig.from_name(
        r=8,
        alpha=8,
        dropout=0.1,
        to_query=True,
        to_value=True,
        to_projection=True,
        **asdict(ours_config),
    )
    initial_lora_model = LoRAGPT(lora_config)
    initial_lora_model.load_state_dict(ours_model.state_dict(), strict=False)

    # save LoRA config file for merge_lora_checkpoint
    with open(lora_config_path, "w") as lora_config_path:
        json.dump({"config": asdict(initial_lora_model.config)}, lora_config_path)

    # mimic that LoRA finetuning only saves lora_A and lora_B
    lora_weights = {k: v for k, v in initial_lora_model.state_dict().items() if "lora" in k}
    torch.save(lora_weights, lora_path)

    # merge and convert the model
    convert_lit_lora_checkpoint(
        model_name=model_name,
        checkpoint_path=ours_ckpt_path,
        lora_path=lora_path,
        merge_lora=True,
        save_merge=True,
    )

    # check that the converted merged model matches when loaded to a HF model
    T = 5
    theirs_config = LlamaConfig(
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=1e-5,
        vocab_size=ours_config.padded_vocab_size,
    )
    theirs_model = LlamaForCausalLM(theirs_config)

    converted_merged_state_dict = torch.load(converted_checkpoint_path)
    theirs_model.load_state_dict(
        converted_merged_state_dict,
        strict=False,  # set to false to ignore self_attn.rotary_emb.inv_freq
        assign=True,
    )

    # check that `W_after = W_initial + (A x B)`
    a = initial_lora_model.transformer.h[0].attn.proj.lora_A
    b = initial_lora_model.transformer.h[0].attn.proj.lora_B
    scaling = initial_lora_model.transformer.h[0].attn.proj.scaling
    delta_w = (b @ a) * scaling
    theirs_model_weight_after = theirs_model.model.layers[0].self_attn.o_proj.weight.clone()
    torch.testing.assert_close(theirs_model_weight_after, initial_weight + delta_w)
