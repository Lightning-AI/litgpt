from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from transformers import LlamaForCausalLM
import torch

from lit_llama.model import LLaMA, LLaMAConfig


def convert_hf_checkpoint(
    model_size: str = "7B",
    hf_checkpoint_path: Path = Path("checkpoints/llama-7b-hf"),
    lit_checkpoint: Path = Path("checkpoints/lit-llama.ckpt"),
    verify: bool = False,
) -> None:
    """
    Perform the reverse operation of: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
    """

    print("Loading weights from pretrained LLaMA %s" % model_size)

    config = LLaMAConfig.from_name(model_size)
    model = LLaMA(config)
    sd = model.state_dict()

    model_hf = LlamaForCausalLM.from_pretrained(hf_checkpoint_path)
    sd_hf = model_hf.state_dict()

    qkv_size = model.transformer.h[0].attn.c_attn.weight.shape[0] // 3
    n_blocks = len(model.transformer.h)

    def permute(w):
        dim = config.n_embd
        return (
            w.view(config.n_head, 2, dim // config.n_head // 2, dim)
            .transpose(1, 2)
            .reshape(dim, dim)
        )

    with torch.no_grad():
        sd["transformer.wte.weight"].copy_(sd_hf["model.embed_tokens.weight"])
        sd["transformer.ln_f.scale"].copy_(sd_hf["model.norm.weight"])
        sd["lm_head.weight"].copy_(sd_hf["lm_head.weight"])

        for i in range(n_blocks):
            sd[f"transformer.h.{i}.attn.c_proj.weight"].copy_(
                sd_hf[f"model.layers.{i}.self_attn.o_proj.weight"]
            )

            sd[f"transformer.h.{i}.attn.c_attn.weight"][:qkv_size] = permute(
                sd_hf[f"model.layers.{i}.self_attn.q_proj.weight"]
            )
            sd[f"transformer.h.{i}.attn.c_attn.weight"][qkv_size:-qkv_size] = permute(
                sd_hf[f"model.layers.{i}.self_attn.k_proj.weight"]
            )
            sd[f"transformer.h.{i}.attn.c_attn.weight"][-qkv_size:] = sd_hf[
                f"model.layers.{i}.self_attn.v_proj.weight"
            ]

            sd[f"transformer.h.{i}.mlp.c_fc1.weight"].copy_(
                sd_hf[f"model.layers.{i}.mlp.gate_proj.weight"]
            )
            sd[f"transformer.h.{i}.mlp.c_fc2.weight"].copy_(
                sd_hf[f"model.layers.{i}.mlp.up_proj.weight"]
            )
            sd[f"transformer.h.{i}.mlp.c_proj.weight"].copy_(
                sd_hf[f"model.layers.{i}.mlp.down_proj.weight"]
            )

            sd[f"transformer.h.{i}.rms_1.scale"].copy_(
                sd_hf[f"model.layers.{i}.input_layernorm.weight"]
            )
            sd[f"transformer.h.{i}.rms_2.scale"].copy_(
                sd_hf[f"model.layers.{i}.post_attention_layernorm.weight"]
            )

    if verify:
        token_sample = torch.randint(
            0, config.vocab_size, size=(1, config.block_size), dtype=torch.int64
        )

        with torch.no_grad():
            out = model(token_sample)
            out_hf = model_hf(token_sample)

        assert torch.allclose(out, out_hf["logits"])

    torch.save(model.state_dict(), lit_checkpoint)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint)
