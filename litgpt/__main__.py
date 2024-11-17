# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import warnings
import torch

from litgpt.chat.base import main as chat_fn
from litgpt.finetune.adapter import setup as finetune_adapter_fn
from litgpt.finetune.adapter_v2 import setup as finetune_adapter_v2_fn
from litgpt.finetune.full import setup as finetune_full_fn
from litgpt.finetune.lora import setup as finetune_lora_fn
from litgpt.generate.adapter import main as generate_adapter_fn
from litgpt.generate.adapter_v2 import main as generate_adapter_v2_fn
from litgpt.generate.base import main as generate_base_fn
from litgpt.generate.full import main as generate_full_fn
from litgpt.generate.sequentially import main as generate_sequentially_fn
from litgpt.generate.tp import main as generate_tp_fn
from litgpt.pretrain import setup as pretrain_fn
from litgpt.scripts.convert_hf_checkpoint import convert_hf_checkpoint as convert_hf_checkpoint_fn
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint as convert_lit_checkpoint_fn
from litgpt.scripts.convert_pretrained_checkpoint import (
    convert_pretrained_checkpoint as convert_pretrained_checkpoint_fn,
)
from litgpt.scripts.download import download_from_hub as download_fn
from litgpt.scripts.merge_lora import merge_lora as merge_lora_fn
from litgpt.eval.evaluate import convert_and_evaluate as evaluate_fn
from litgpt.deploy.serve import run_server as serve_fn
from jsonargparse import set_config_read_mode, set_docstring_parse_options, CLI


def main() -> None:
    parser_data = {
        "download": download_fn,
        "chat": chat_fn,
        "finetune": finetune_lora_fn,
        "finetune_lora": finetune_lora_fn,
        "finetune_full": finetune_full_fn,
        "finetune_adapter": finetune_adapter_fn,
        "finetune_adapter_v2": finetune_adapter_v2_fn,
        "pretrain": pretrain_fn,
        "generate": generate_base_fn,
        "generate_full": generate_full_fn,
        "generate_adapter": generate_adapter_fn,
        "generate_adapter_v2": generate_adapter_v2_fn,
        "generate_sequentially": generate_sequentially_fn,
        "generate_tp": generate_tp_fn,
        "convert_to_litgpt": convert_hf_checkpoint_fn,
        "convert_from_litgpt": convert_lit_checkpoint_fn,
        "convert_pretrained_checkpoint": convert_pretrained_checkpoint_fn,
        "merge_lora": merge_lora_fn,
        "evaluate": evaluate_fn,
        "serve": serve_fn
    }

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    # PyTorch bug that raises a false-positive warning
    # More info: https://github.com/Lightning-AI/litgpt/issues/1561
    warning_message = (
        r"The epoch parameter in `scheduler.step\(\)` was not necessary and is being deprecated.*"
    )

    warnings.filterwarnings(
        action="ignore",
        message=warning_message,
        category=UserWarning,
        module=r'.*torch\.optim\.lr_scheduler.*'
    )

    torch.set_float32_matmul_precision("high")
    CLI(parser_data)


if __name__ == "__main__":
    main()
