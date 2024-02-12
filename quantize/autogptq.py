import importlib
import json
import sys
import time
from functools import reduce
from pathlib import Path
from typing import Any, Literal, Optional, Union

import lightning as L
import torch
from auto_gptq.modeling._base import BaseGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.modeling._utils import autogptq_post_init
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
# These classes have changed `forward` methods, so they can accept and provide such arguments as `attention_mask`
# to "behave" like HuggingFace models and thus become compatible with AutoGPTQ.
class GPTForAutoGPTQ(GPT):
    # Changes to make it compatible with AutoGPTQ:
    # - "input_ids" instead of "idx"
    # - **kwargs store "attention_mask" (that we don't need), but AutoGPTQ provides
    def forward(self, input_ids: torch.Tensor, input_pos: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        """For AutoGPTQ this forward pass is needed only to capture arguments (through a forward hook attached to the
        first layer) that will be send to each layer (Transformer block). That means it will be called only once."""

        T = input_ids.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")
        x = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)

        # - LayerHijacker (inside the quantize method) doesn't expect all these arguments, but expects kwargs,
        #   so we can provide them as keyword arguments
        self.transformer.h[0](x, cos=self.cos[:T], sin=self.sin[:T], mask=None, input_pos=input_pos, **kwargs)


class BlockForAutoGPTQ(Block):
    # Changes to make it compatible with AutoGPTQ:
    # - accept "attention_mask" that is provided by AutoGPTQ, but don't use it
    def forward(self, *args: Any, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> tuple[torch.Tensor]:
        output = super().forward(*args, **kwargs)
        # HF TransformerBlock returns at least (hidden_state,)
        return (output,)


class AutoGPTQ(BaseGPTQForCausalLM):

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

        # TODO: rephrase it
        # chained attribute name of transformer layer block
        self.layers_block_name = "transformer.h"

        # TODO: rephrase it
        # chained attribute names of other nn modules that in the same level as the transformer layer block
        # but are called before it
        # (aren't quantized)
        self.outside_layer_modules = ["transformer.wte"]

        # TODO: rephrase it
        # chained attribute names of linear layers in a transformer layer module
        # normally, there are four sub lists, for each one the modules in it can be seen as one operation,
        # and the order should be the order when they are truly executed, in this case (and usually in most cases),
        # they are: attention q_k_v projection, attention output projection, MLP project input, MLP project output
        self.inside_layer_modules = [
            ["attn.attn"],
            ["attn.proj"],
        ]

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
                    ["mlp.fc_1", "mlp.fc_2"],
                    ["mlp.proj"],
                ]
            )
        elif mlp_class == "LLaMAMoE":
            # AutoGPTQ doesn't quantize "gate" layer
            # [
            #   [mlp.experts.0.fc_1, ..., mlp.experts.0.fc2, ...],
            #   [mlp.experts.0.proj],
            # ]
            self.inside_layer_modules.extend(
                [
                    [
                        f"mlp.experts.{expert_idx}.{attr}"
                        for attr in ("fc_1", "fc_2")
                        for expert_idx in range(self.model.config.n_expert)
                    ],
                    [f"mlp.experts.{expert_idx}.proj" for expert_idx in range(self.model.config.n_expert)],
                ]
            )
        else:
            raise ValueError(f"MLP class `{mlp_class}` is not yet supported by AutoGPTQ")

        self.lm_head_name = "lm_head"

    def convert_model_to_quantized(self, kernel: Literal["cuda", "exllama", "exllamav2", "triton"]) -> None:
        # TODO: add docstring

        # Kernel    Supported precisions
        #           ┌───────────────┐
        #           ┆ 2 ┆ 3 ┆ 4 ┆ 8 ┆
        #           ├---------------┤
        # cuda      ┆ ✔ ┆ ✔ ┆ ✔ ┆ ✔ ┆
        # exllama   ┆   ┆   ┆ ✔ ┆   ┆
        # exllamav2 ┆   ┆   ┆ ✔ ┆   ┆
        # triton    ┆ ✔ ┆   ┆ ✔ ┆ ✔ ┆
        #           └───────────────┘

        if kernel not in ("cuda", "exllama", "exllamav2", "triton"):
            raise ValueError(f"Kernel `{kernel}` is not supported.")

        # Check if the kernel is compatible with the precision used for quantization
        if (kernel == "triton" and self.quantize_config.bits == 3) or (
            kernel in ("exllama", "exllamav2") and self.quantize_config.bits != 4
        ):
            raise ValueError(f"Kernel '{kernel}' doesn't support {self.quantize_config.bits}bit precision. ")

        QuantLinear = importlib.import_module(f"auto_gptq.nn_modules.qlinear.qlinear_{kernel}").QuantLinear

        inside_layer_modules = tuple(sum(self.inside_layer_modules, []))

        for name, module in list(self.model.named_modules()):
            if isinstance(module, torch.nn.Linear) and name.endswith(inside_layer_modules):
                parent_module = reduce(getattr, name.split(".")[:-1], self.model)
                attribute = name.split(".")[-1]
                new_layer = QuantLinear(
                    infeatures=module.in_features,
                    outfeatures=module.out_features,
                    bits=self.quantize_config.bits,
                    group_size=self.quantize_config.group_size,
                    bias=True,  # AutoGPTQ always creates bias during quantization
                    weight_dtype=module.weight.dtype,
                )
                device = next(module.parameters()).device
                new_layer = new_layer.to(device)
                parent_module.__setattr__(attribute, new_layer)

    def post_init(self, max_input_length: Union[None, int] = None) -> None:
        # TODO: add docstring
        autogptq_post_init(self.model, use_act_order=self.quantize_config.desc_act, max_input_length=max_input_length)

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
    use_triton: bool = False,
) -> None:
    # TODO: add a docstring and/or add info into a README file

    check_valid_checkpoint_dir(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    if output_path is None:
        output_path = checkpoint_dir / "lit_model_gptq.4bit.pth"

    # --- Load and prepare calibration data ---
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
    # "torch.float16" precision is preferred during the inference, so it's better to
    # have the model in this precision during the quantization.
    fabric = L.Fabric(accelerator="cpu", precision="16-true")
    with fabric.init_module(empty_init=False), _ClassReplacementContextManager(
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
    autogptq = AutoGPTQ(model, quantized=False, quantize_config=quantize_config)

    quantize_time = time.perf_counter()
    autogptq.quantize(calibration_data, batch_size=batch_size, use_triton=use_triton)
    fabric.print(f"Quantization time: {(time.perf_counter()-quantize_time):.2f}s")

    # Save the model
    # TODO: AutoGPTQ creates bias weights even if they are not needed.
    # Trim them before saving (per config)
    torch.save(autogptq.model.state_dict(), output_path)

    # Save quantize config with the used kernel - will be reused during inference
    quantize_config.kernel = next(
        module.QUANT_TYPE for module in autogptq.model.modules() if hasattr(module, "QUANT_TYPE")
    )
    with open(output_path.with_name("autogptq_config.json"), "w") as fp:
        json.dump(quantize_config.__dict__, fp)


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
