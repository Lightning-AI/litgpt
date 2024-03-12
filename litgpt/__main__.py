# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch


def main():
    from jsonargparse import ArgumentParser

    from litgpt.pretrain import setup as pretrain_fn
    from litgpt.chat.base import main as chat_fn
    from litgpt.scripts.download import download as download_fn
    from litgpt.scripts.merge_lora import merge_lora as merge_lora_fn
    from litgpt.finetune.lora import setup as finetune_lora_fn
    from litgpt.finetune.full import setup as finetune_full_fn
    from litgpt.finetune.adapter import setup as finetune_adapter_fn
    from litgpt.finetune.adapter_v2 import setup as finetune_adapter_v2_fn
    from litgpt.generate.base import main as generate_base_fn
    from litgpt.generate.full import main as generate_full_fn
    from litgpt.generate.lora import main as generate_lora_fn
    from litgpt.generate.adapter import main as generate_adapter_fn
    from litgpt.generate.adapter_v2 import main as generate_adapter_v2_fn
    from litgpt.generate.sequentially import main as generate_sequentially_fn
    from litgpt.generate.tp import main as generate_tp_fn
    from litgpt.scripts.convert_pretrained_checkpoint import (
        convert_pretrained_checkpoint as convert_pretrained_checkpoint_fn,
    )
    from litgpt.scripts.convert_hf_checkpoint import convert_hf_checkpoint as convert_hf_checkpoint_fn
    from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint as convert_lit_checkpoint_fn

    parser_data = {
        "pretrain": {"help": "Pretrain a model.", "fn": pretrain_fn},
        "finetune": {
            "help": "Finetune a model with one of our existing methods.",
            "lora": {"help": "Finetune a model with LoRA.", "fn": finetune_lora_fn},
            "full": {"help": "Finetune a model.", "fn": finetune_full_fn},
            "adapter": {"help": "Finetune a model with Adapter.", "fn": finetune_adapter_fn},
            "adapter_v2": {"help": "Finetune a model with Adapter v2.", "fn": finetune_adapter_v2_fn},
        },
        "generate": {
            "help": "Generate text samples based on a model and tokenizer.",
            "base": {"fn": generate_base_fn, "help": "Default generation option."},
            "full": {"fn": generate_full_fn, "help": "For models finetuned with `litgpt finetune full`."},
            "lora": {"fn": generate_lora_fn, "help": "For models finetuned with `litgpt finetune lora`."},
            "adapter": {"fn": generate_adapter_fn, "help": "For models finetuned with `litgpt finetune adapter`."},
            "adapter_v2": {
                "fn": generate_adapter_v2_fn,
                "help": "For models finetuned with `litgpt finetune adapter v2`.",
            },
            "sequentially": {
                "fn": generate_sequentially_fn,
                "help": "Generation script that partitions layers across devices to be run sequentially.",
            },
            "tp": {
                "fn": generate_tp_fn,
                "help": "Generation script that uses tensor parallelism to run across devices.",
            },
        },
        "chat": {"help": "Chat with a model.", "fn": chat_fn},
        "convert": {
            "help": "Utilities to convert from and to LitGPT.",
            "to_litgpt": {"fn": convert_hf_checkpoint_fn, "help": "Convert Hugging Face weights to LitGPT weights."},
            "from_litgpt": {"fn": convert_lit_checkpoint_fn, "help": "Convert LitGPT weights to Hugging Face weights."},
            "pretrained_checkpoint": {
                "fn": convert_pretrained_checkpoint_fn,
                "help": "Convert a checkpoint after pretraining.",
            },
        },
        "download": {"help": "Download weights or tokenizer data from the Hugging Face Hub.", "fn": download_fn},
        # TODO: should this be at the first level?
        "merge_lora": {"help": "Merges the LoRA weights with the base model.", "fn": merge_lora_fn},
    }

    root_parser = ArgumentParser(prog="litgpt")

    # register level 1 subcommands and level 2 subsubcommands. If there are more levels in the future we would want to
    # refactor this to do BFS traversal for registration
    subcommands = root_parser.add_subcommands()
    subcommand_to_parser = {}
    for k, v in parser_data.items():
        subcommand_parser = ArgumentParser()
        if "fn" in v:
            subcommand_parser.add_function_arguments(v["fn"])
        else:
            subcommand_to_parser[k] = subcommand_parser
        subcommands.add_subcommand(k, subcommand_parser, help=v["help"])
    for subcommand, parser in subcommand_to_parser.items():
        subcommands = parser.add_subcommands()
        for k, v in parser_data[subcommand].items():
            if k == "help":
                continue
            subsubcommand_parser = ArgumentParser()
            subsubcommand_parser.add_function_arguments(v["fn"])
            subcommands.add_subcommand(k, subsubcommand_parser, help=v["help"])

    args = root_parser.instantiate_classes(root_parser.parse_args())

    subcommand = args.get("subcommand")
    subargs = args.get(subcommand)
    subsubcommand = subargs.get("subcommand")
    subsubargs = subargs.get(subsubcommand) if isinstance(subsubcommand, str) else None

    level_1 = parser_data[subcommand]
    if subsubcommand is None:
        fn = level_1["fn"]
        kwargs = subargs
    else:
        fn = level_1[subsubcommand]["fn"]
        kwargs = subsubargs

    torch.set_float32_matmul_precision("high")

    fn(**kwargs)


if __name__ == "__main__":
    main()
