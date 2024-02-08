import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

import lightning as L
import torch
from auto_gptq.modeling._base import BaseGPTQForCausalLM, BaseQuantizeConfig
from lightning.fabric.plugins.precision.utils import _ClassReplacementContextManager

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config
from lit_gpt.model import Block
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    load_checkpoint,
)


# `GPTForAutoGPTQ` and `BlockForAutoGPTQ` are only needed for quantization process.
# These classes have slightly changed `forward` methods, so they can accept and provide such arguments as `attention_mask`,
# so they "behave" like HuggingFace models and thus become compatible with AutoGPTQ.
class GPTForAutoGPTQ(GPT):
    # Changes to make it compatible with AutoGPTQ:
    # - "input_ids" instead of "idx"
    # - **kwargs store "attention_mask" (that we don't need), but AutoGPTQ provides
    def forward(self, input_ids: torch.Tensor, input_pos: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        T = input_ids.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            # Changes to make it compatible with AutoGPTQ:
            # - LayerHijacker (inside the quantize method) doesn't expect all these arguments, but expects kwargs,
            #   so we can provide them as keyword arguments
            x = block(x, cos=cos, sin=sin, mask=mask, input_pos=input_pos, **kwargs)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)


class BlockForAutoGPTQ(Block):
    # Changes to make it compatible with AutoGPTQ:
    # - accept "attention_mask" that is provided by AutoGPTQ, but don't use it
    def forward(self, *args: Any, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> tuple[torch.Tensor]:
        output = super().forward(*args, **kwargs)
        # HF TransformerBlock returns at least (hidden_state,)
        return (output,)


class AutoGPTQ(BaseGPTQForCausalLM):
    # TODO: rephrase it
    # chained attribute name of transformer layer block
    layers_block_name = "transformer.h"

    # TODO: rephrase it
    # chained attribute names of other nn modules that in the same level as the transformer layer block
    # but are called before it
    # (aren't quantized)
    outside_layer_modules = ["transformer.wte"]

    # TODO: rephrase it
    # chained attribute names of linear layers in a transformer layer module
    # normally, there are four sub lists, for each one the modules in it can be seen as one operation,
    # and the order should be the order when they are truly executed, in this case (and usually in most cases),
    # they are: attention q_k_v projection, attention output projection, MLP project input, MLP project output
    inside_layer_modules = [
        ["attn.attn"],
        ["attn.proj"],
    ]

    lm_head_name = "lm_head"

    def __init__(
        self,
        model: GPTForAutoGPTQ,
        quantized: bool,
        quantize_config: BaseQuantizeConfig,
        is_triton_backend: bool = False,
        injected_fused_attention: bool = False,
        injected_fused_mlp: bool = False,
        trainable: bool = False,
    ):
        super().__init__(
            model,
            quantized,
            quantize_config,
            is_triton_backend,
            injected_fused_attention,
            injected_fused_mlp,
            trainable,
        )
        # TODO: add docstring
        # NOTE: `is_triton_backend` is used only to tell that's it's possible to train only with triton
        # NOTE: `injected_...` are used for peft

        mlp_class = self.model.config._mlp_class
        if mlp_class == "GptNeoxMLP":
            self.inside_layer_modules.extend(
                [
                    ["mlp.fc"],
                    ["mlp.proj"],
                ]
            )
        elif mlp_class == "LLaMAMLP":
            self.inside_layer_modules.extend(
                [
                    ["mlp.fc_1", "mlp.fc_2"],  # gate, up
                    ["mlp.proj"],  # down
                ]
            )
        elif mlp_class == "LLaMAMoE":
            raise NotImplementedError("LLaMAMoE is not yet supported")
        else:
            raise ValueError(f"MLP class `{mlp_class}` is not yet supported by AutoGPTQ")

    # in AutoGPTQ method `to` only expects `device`
    def to(self, device: Union[str, torch.device], dtype: torch.dtype) -> "AutoGPTQ":
        self.model.to(device=device, dtype=dtype)
        return self


def main(
    *,
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    output_path: Optional[Path] = None,
    bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.01,
    desc_act: bool = True,
    static_groups: bool = False,
    sym: bool = True,
    true_sequential: bool = True,
    n_samples: int = 1024,
    batch_size: int = 32,
) -> None:
    # TODO: add a docstring and/or add info into a README file

    check_valid_checkpoint_dir(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    if output_path is None:
        output_path = checkpoint_dir / "lit_model_gptq.4bit.pth"

    # --- Load and prepare a calibraion data ---
    calibration_data = torch.load(data_dir / "test.pt")
    # AutoGPTQ expects a list of dicts with two keys in each: "input_ids" and "attention_mask".
    # Since Lit model doesn't need "attention_mask", we can "fake" it.
    calibration_data = [
        {"input_ids": record["input_ids"], "attention_mask": [1] * len(record["input_ids"])}
        for record in calibration_data[:n_samples]
    ]

    # --- Create a model ---
    config = Config.from_json(checkpoint_dir / "lit_config.json")
    # The model is loaded into a CPU RAM and each layer, that is about to be quantized,
    # is moved to a GPU by the AutoGPTQ itself.
    # "torch.float16" precision is prefered during the inference, so it's better to
    # have the model in this precision during the quantization.
    fabric = L.Fabric(accelerator="cpu", precision="16-true")
    with fabric.init_module(empty_init=True), _ClassReplacementContextManager(
        {"lit_gpt.model.Block": BlockForAutoGPTQ}
    ):
        model = GPTForAutoGPTQ(config)
    model.eval()
    load_checkpoint(fabric, model, checkpoint_path)

    # --- Update model's config ---
    # AutoGPTQ wants to retrieve these 4 parameters, but Lit's config doesn't have
    model.config.model_type = None  # used in .from_pretrained and .from_quantized
    model.config.pad_token_id = None  # ._prepare_examples_for_quantization
    model.config.eos_token_id = Tokenizer(checkpoint_dir).eos_id  # _prepare_examples_for_quantization
    model.config.use_cache = False  # for quantization it's disabled anyway

    # --- Quantize the model ---
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=desc_act,
        static_groups=static_groups,
        sym=sym,
        true_sequential=true_sequential,
    )
    model = AutoGPTQ(model, quantized=False, quantize_config=quantize_config)

    quantize_time = time.perf_counter()
    model.quantize(calibration_data, batch_size=batch_size)
    fabric.print(f"Quantization time: {(time.perf_counter()-quantize_time):.2f}s")

    # Save model
    # TODO: AutoGPTQ creates bias weights even if they are not needed.
    # Trim them before saving (per config)
    torch.save(model.model.state_dict(), output_path)


if __name__ == "__main__":

    import logging

    from jsonargparse import CLI

    # AutoGPTQ's quantization logs
    logging.basicConfig(
        format="%(asctime)s ｜ %(levelname)s ｜ %(name)-27s ｜ %(message)s", datefmt="%Y-%m-%d ｜ %H:%M:%S"
    )
    logging.getLogger("auto_gptq").setLevel(logging.INFO)

    torch.set_float32_matmul_precision("high")

    CLI(main)
