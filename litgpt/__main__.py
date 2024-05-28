# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from typing import TYPE_CHECKING, Any

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


if TYPE_CHECKING:
    from jsonargparse import ArgumentParser


def _new_parser(**kwargs: Any) -> "ArgumentParser":
    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(**kwargs)
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    return parser


def main() -> None:
    parser_data = {
        "download": {"fn": download_fn, "_help": "Download weights or tokenizer data from the Hugging Face Hub."},
        "chat": {"fn": chat_fn, "_help": "Chat with a model."},

        "finetune": {"fn": finetune_lora_fn, "_help": "Finetune a model (uses LoRA)."},
        "finetune_lora": {"fn": finetune_lora_fn, "_help": "Finetune a model with LoRA."},
        "finetune_full": {"fn": finetune_full_fn, "_help": "Finetune a model."},
        "finetune_adapter": {"fn": finetune_adapter_fn, "_help": "Finetune a model with Adapter."},
        "finetune_adapter_v2": {"fn": finetune_adapter_v2_fn, "_help": "Finetune a model with Adapter v2."},

        "pretrain": {"fn": pretrain_fn, "_help": "Pretrain a model.", },

        "generate": {"fn": generate_base_fn, "_help": "Default generation option."},
        "generate_full": {"fn": generate_full_fn, "_help": "For models finetuned with `litgpt_finetune_full"},
        "generate_adapter": {"fn": generate_adapter_fn, "_help": "For models finetuned with `litgpt_finetune_adapter`."},
        "generate_adapter_v2": {"fn": generate_adapter_v2_fn, "_help": "For models finetuned with `litgpt finetune_adapter_v2`."},
        "generate_sequentially": {"fn": generate_sequentially_fn, "_help": "Generation script that partitions layers across devices to be run sequentially."},
        "generate_tp": {"fn": generate_tp_fn, "_help": "Generation script that uses tensor parallelism to run across devices."},

        "convert_to_litgpt": {"fn": convert_hf_checkpoint_fn, "_help": "Convert Hugging Face weights to LitGPT weights."},
        "convert_from_litgpt": {"fn": convert_lit_checkpoint_fn, "_help": "Convert LitGPT weights to Hugging Face weights."},
        "convert_pretrained_checkpoint": {"fn": convert_pretrained_checkpoint_fn, "_help": "Convert a checkpoint after pretraining."},

        "merge_lora": {"fn": merge_lora_fn, "_help": "Merges the LoRA weights with the base model."},
        "evaluate": {"fn": evaluate_fn, "_help": "Evaluate a model with the LM Evaluation Harness."},
        "serve": {"fn": serve_fn, "_help": "Serve and deploy a model with LitServe."},
    }

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

    from jsonargparse import set_config_read_mode, set_docstring_parse_options, CLI

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    CLI(parser_data)
    torch.set_float32_matmul_precision("high")



if __name__ == "__main__":
    main()