# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import sys

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
        "download": {"help": "Download weights or tokenizer data from the Hugging Face Hub.", "fn": download_fn},
        "chat": {"help": "Chat with a model.", "fn": chat_fn},

        "finetune": {"help": "Finetune a model (uses LoRA).", "fn": finetune_lora_fn},
        "finetune_lora": {"help": "Finetune a model with LoRA.", "fn": finetune_lora_fn},
        "finetune_full": {"help": "Finetune a model.", "fn": finetune_full_fn},
        "finetune_adapter": {"help": "Finetune a model with Adapter.", "fn": finetune_adapter_fn},
        "finetune_adapter_v2": {"help": "Finetune a model with Adapter v2.", "fn": finetune_adapter_v2_fn},

        "pretrain": {"help": "Pretrain a model.", "fn": pretrain_fn},

        "generate": {"fn": generate_base_fn, "help": "Default generation option."},
        "generate_full": {"fn": generate_full_fn, "help": "For models finetuned with `litgpt_finetune_full"},
        "generate_adapter": {"fn": generate_adapter_fn, "help": "For models finetuned with `litgpt_finetune_adapter`."},
        "generate_adapter_v2": {"fn": generate_adapter_v2_fn, "help": "For models finetuned with `litgpt finetune_adapter_v2`."},
        "generate_sequentially": {"fn": generate_sequentially_fn, "help": "Generation script that partitions layers across devices to be run sequentially."},
        "generate_tp": {"fn": generate_tp_fn, "help": "Generation script that uses tensor parallelism to run across devices."},

        "convert_to_litgpt": {"fn": convert_hf_checkpoint_fn, "help": "Convert Hugging Face weights to LitGPT weights."},
        "convert_from_litgpt": {"fn": convert_lit_checkpoint_fn, "help": "Convert LitGPT weights to Hugging Face weights."},
        "convert_pretrained_checkpoint": {"fn": convert_pretrained_checkpoint_fn, "help": "Convert a checkpoint after pretraining."},

        "merge_lora": {"help": "Merges the LoRA weights with the base model.", "fn": merge_lora_fn},
        "evaluate": {"help": "Evaluate a model with the LM Evaluation Harness.", "fn": evaluate_fn},
        "serve": {"help": "Serve and deploy a model with LitServe.", "fn": serve_fn},
    }

    from jsonargparse import set_config_read_mode, set_docstring_parse_options

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    root_parser = _new_parser(prog="litgpt")

    # register level 1 subcommands
    subcommands = root_parser.add_subcommands()
    for k, v in parser_data.items():
        subcommand_parser = _new_parser()
        if "fn" in v:
            skip_args = {"optimizer"} if k in ("finetune", "pretrain") else {}
            subcommand_parser.add_function_arguments(v["fn"], skip=skip_args)
        if k in ("finetune", "pretrain"):
            subcommand_parser.add_subclass_arguments(
                torch.optim.Optimizer, "optimizer", instantiate=False, fail_untyped=False, skip={"params"}
            )
            subcommand_parser.set_defaults({"optimizer": "AdamW"})
        subcommands.add_subcommand(k, subcommand_parser, help=v["help"])

    args = root_parser.parse_args()
    args = root_parser.instantiate_classes(args)

    subcommand = args.get("subcommand")
    subargs = args.get(subcommand)

    level_1 = parser_data[subcommand]
    fn = level_1["fn"]
    kwargs = subargs
    kwargs.pop("config")

    torch.set_float32_matmul_precision("high")

    # dictionary unpacking on the jsonargparse namespace seems to flatten inner namespaces. I dont know if that's a bug or intended
    # but we can simply convert to dict at this point
    kwargs = kwargs.as_dict()

    fn(**kwargs)


if __name__ == "__main__":
    main()
