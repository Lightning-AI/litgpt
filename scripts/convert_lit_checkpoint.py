import contextlib
import gc
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.utils import NotYetLoadedTensor, incremental_save, lazy_load
from scripts.convert_hf_checkpoint import layer_template, load_param


def copy_weights_falcon(
    size: Literal["7b", "40b"],
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    weight_map = {
        "transformer.wte.weight": "transformer.word_embeddings.weight",
        "transformer.h.{}.attn.attn.weight": "transformer.h.{}.self_attention.query_key_value.weight",
        "transformer.h.{}.attn.proj.weight": "transformer.h.{}.self_attention.dense.weight",
        "transformer.h.{}.mlp.fc.weight": "transformer.h.{}.mlp.dense_h_to_4h.weight",
        "transformer.h.{}.mlp.proj.weight": "transformer.h.{}.mlp.dense_4h_to_h.weight",
        "transformer.ln_f.bias": "transformer.ln_f.bias",
        "transformer.ln_f.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }
    # the original model definition is different for each size
    if size == "7b":
        weight_map.update(
            {
                "transformer.h.{}.norm_1.bias": "transformer.h.{}.input_layernorm.bias",
                "transformer.h.{}.norm_1.weight": "transformer.h.{}.input_layernorm.weight",
            }
        )
    elif size == "40b":
        weight_map.update(
            {
                "transformer.h.{}.norm_1.bias": "transformer.h.{}.ln_attn.bias",
                "transformer.h.{}.norm_1.weight": "transformer.h.{}.ln_attn.weight",
                "transformer.h.{}.norm_2.bias": "transformer.h.{}.ln_mlp.bias",
                "transformer.h.{}.norm_2.weight": "transformer.h.{}.ln_mlp.weight",
            }
        )
    else:
        raise NotImplementedError

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, None)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_gpt_neox(
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "gpt_neox.embed_in.weight",
        "transformer.h.{}.norm_1.bias": "gpt_neox.layers.{}.input_layernorm.bias",
        "transformer.h.{}.norm_1.weight": "gpt_neox.layers.{}.input_layernorm.weight",
        "transformer.h.{}.attn.attn.bias": "gpt_neox.layers.{}.attention.query_key_value.bias",
        "transformer.h.{}.attn.attn.weight": "gpt_neox.layers.{}.attention.query_key_value.weight",
        "transformer.h.{}.attn.proj.bias": "gpt_neox.layers.{}.attention.dense.bias",
        "transformer.h.{}.attn.proj.weight": "gpt_neox.layers.{}.attention.dense.weight",
        "transformer.h.{}.norm_2.bias": "gpt_neox.layers.{}.post_attention_layernorm.bias",
        "transformer.h.{}.norm_2.weight": "gpt_neox.layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.mlp.fc.bias": "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias",
        "transformer.h.{}.mlp.fc.weight": "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight",
        "transformer.h.{}.mlp.proj.bias": "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias",
        "transformer.h.{}.mlp.proj.weight": "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight",
        "transformer.ln_f.bias": "gpt_neox.final_layer_norm.bias",
        "transformer.ln_f.weight": "gpt_neox.final_layer_norm.weight",
        "lm_head.weight": "embed_out.weight",
    }

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, None)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_llama(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    weight_map = {
        "transformer.wte.weight": "model.embed_tokens.weight",
        "transformer.h.{}.norm_1.weight": "model.layers.{}.input_layernorm.weight",
        "transformer.h.{}.attn.proj.weight": "model.layers.{}.self_attn.o_proj.weight",
        "transformer.h.{}.norm_2.weight": "model.layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.mlp.fc_1.weight": "model.layers.{}.mlp.gate_proj.weight",
        "transformer.h.{}.mlp.fc_2.weight": "model.layers.{}.mlp.up_proj.weight",
        "transformer.h.{}.mlp.proj.weight": "model.layers.{}.mlp.down_proj.weight",
        "transformer.ln_f.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for name, param in lit_weights.items():
        if name.endswith(".attn.attn.weight"):
            from_name, number = layer_template(name, 2)
            q = "model.layers.{}.self_attn.q_proj.weight".format(number)
            k = "model.layers.{}.self_attn.k_proj.weight".format(number)
            v = "model.layers.{}.self_attn.v_proj.weight".format(number)
            qkv = load_param(param, name, None)
            qp, kp, vp = tensor_split(qkv, config)
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param
        elif "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(number)
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param

        else:
            to_name = weight_map[name]
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param


def tensor_split(
    param: Union[torch.Tensor, NotYetLoadedTensor], config: Config
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def kstart(start, blen, klen) -> int:
        """returns start index of keys in batch"""
        return start + (blen - (klen * 2))

    def vstart(start, blen, klen) -> int:
        """returns start index of values in batch"""
        return start + blen - klen

    def vend(start, blen) -> int:
        """returns last index of values in batch"""
        return start + blen

    # num observations
    nobs = param.shape[0]
    # batch length
    blen = nobs // config.n_query_groups
    # key length in batch
    klen = config.head_size
    # value length in batch
    vlen = config.head_size
    # the starting index of each new batch
    starts = range(0, nobs, blen)
    # the indices to splice on
    splices = [(s, kstart(s, blen, klen), vstart(s, blen, vlen), vend(s, blen)) for s in starts]

    qc = ()
    kc = ()
    vc = ()

    for splice in splices:
        qs, ks, vs, ve = splice
        qc += (param[qs:ks, :],)
        kc += (param[ks:vs, :],)
        vc += (param[vs:ve, :],)

    q = torch.cat(qc)
    k = torch.cat(kc)
    v = torch.cat(vc)

    return q, k, v


def maybe_unwrap_state_dict(lit_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return lit_weights.get("model", lit_weights)


def check_conversion_supported(lit_weights: Dict[str, torch.Tensor]) -> None:
    weight_names = {wk.split(".")[-1] for wk in lit_weights}
    # LoRA or QLoRA
    if any("lora" in wn for wn in weight_names):
        raise ValueError("Model weights must be merged using `lora.merge_lora_weights()` before conversion.")
    # adapter v2. adapter_bias will only be in adapter_v2
    elif "adapter_bias" in weight_names:
        raise NotImplementedError("Converting models finetuned with adapter_v2 not yet supported.")
    # adapter. gating_factor is in adapter and adapter_v2
    elif "gating_factor" in weight_names:
        raise NotImplementedError("Converting models finetuned with adapter not yet supported.")


@torch.inference_mode()
def convert_lit_checkpoint(*, checkpoint_name: str, out_dir: Path, model_name: str) -> None:
    config = Config.from_name(model_name)

    if "falcon" in model_name:
        copy_fn = partial(copy_weights_falcon, "40b" if config.n_embd == 8192 else "7b")
    elif config._mlp_class == "LLaMAMLP":
        copy_fn = partial(copy_weights_llama, config)
    else:
        copy_fn = copy_weights_gpt_neox

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # checkpoint_name cannot be hardcoded because there exists different outputs such as
    # ("lit_model_finetuned.pth", "lit_model_lora_finetuned.pth", "lit_model_adapter_finetuned.pth"")
    pth_file = out_dir / checkpoint_name
    bin_file = pth_file.with_suffix(".bin")

    with incremental_save(bin_file) as saver:
        with contextlib.ExitStack() as stack:
            lit_weights = stack.enter_context(lazy_load(pth_file))
            lit_weights = maybe_unwrap_state_dict(lit_weights)
            check_conversion_supported(lit_weights)
            copy_fn(sd, lit_weights, saver=saver)
            gc.collect()
        saver.save(sd)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_lit_checkpoint, as_positional=False)
