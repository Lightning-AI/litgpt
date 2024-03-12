# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch


def main():
    from jsonargparse import ArgumentParser, Namespace

    # root parser
    parser = ArgumentParser(prog="litgpt")

    # level 1 subcommands

    pretrain = ArgumentParser()
    from litgpt.pretrain import setup as pretrain_fn
    pretrain.add_function_arguments(pretrain_fn)

    finetune = ArgumentParser()

    generate = ArgumentParser()

    chat = ArgumentParser()
    from litgpt.chat.base import main as chat_fn
    chat.add_function_arguments(chat_fn)

    convert = ArgumentParser()

    download = ArgumentParser()
    from litgpt.scripts.download import download as download_fn
    download.add_function_arguments(download_fn)

    merge_lora = ArgumentParser()
    from litgpt.scripts.merge_lora import merge_lora as merge_lora_fn
    merge_lora.add_function_arguments(merge_lora_fn)

    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("pretrain", pretrain, help="Pretrain a model.")
    subcommands.add_subcommand("finetune", finetune, help="Finetune a model with one of our existing methods.")
    subcommands.add_subcommand("generate", generate, help="Generate text samples based on a model and tokenizer.")
    subcommands.add_subcommand("chat", chat, help="Chat with a model.")
    subcommands.add_subcommand("convert", convert, help="Utilities to convert from and to LitGPT.")
    subcommands.add_subcommand("download", download, help="Download weights or tokenizer data from the Hugging Face Hub.")
    # TODO: should this be at the first level?
    subcommands.add_subcommand("merge_lora", merge_lora, help="Merges the LoRA weights with the base model.")

    # level 2 subcommands

    finetune_subcommands = finetune.add_subcommands()
    finetune_lora = ArgumentParser()
    finetune_full = ArgumentParser()
    finetune_adapter = ArgumentParser()
    finetune_adapter_v2 = ArgumentParser()
    from litgpt.finetune.lora import setup as finetune_lora_fn
    from litgpt.finetune.full import setup as finetune_full_fn
    from litgpt.finetune.adapter import setup as finetune_adapter_vn
    from litgpt.finetune.adapter_v2 import setup as finetune_adapter_v2_fn
    finetune_lora.add_function_arguments(finetune_lora_fn)
    finetune_full.add_function_arguments(finetune_full_fn)
    finetune_adapter.add_function_arguments(finetune_adapter_vn)
    finetune_adapter_v2.add_function_arguments(finetune_adapter_v2_fn)
    finetune_subcommands.add_subcommand("lora", finetune_lora, help="Finetune a model with LoRA.")
    finetune_subcommands.add_subcommand("full", finetune_full, help="Finetune a model.")
    finetune_subcommands.add_subcommand("adapter", finetune_adapter, help="Finetune a model with Adapter.")
    finetune_subcommands.add_subcommand("adapter_v2", finetune_adapter_v2, help="Finetune a model with Adapter v2.")

    generate_subcommands = generate.add_subcommands()
    generate_base = ArgumentParser()
    generate_full = ArgumentParser()
    generate_lora = ArgumentParser()
    generate_adapter = ArgumentParser()
    generate_adapter_v2 = ArgumentParser()
    generate_sequentially = ArgumentParser()
    generate_tp = ArgumentParser()
    from litgpt.generate.base import main as generate_base_fn
    from litgpt.generate.full import main as generate_full_fn
    from litgpt.generate.lora import main as generate_lora_fn
    from litgpt.generate.adapter import main as generate_adapter_fn
    from litgpt.generate.adapter_v2 import main as generate_adapter_v2_fn
    from litgpt.generate.sequentially import main as generate_sequentially_fn
    from litgpt.generate.tp import main as generate_tp_fn
    generate_base.add_function_arguments(generate_base_fn)
    generate_full.add_function_arguments(generate_full_fn)
    generate_lora.add_function_arguments(generate_lora_fn)
    generate_adapter.add_function_arguments(generate_adapter_fn)
    generate_adapter_v2.add_function_arguments(generate_adapter_v2_fn)
    generate_sequentially.add_function_arguments(generate_sequentially_fn)
    generate_tp.add_function_arguments(generate_tp_fn)
    generate_subcommands.add_subcommand("base", generate_base, help="Default generation option.")
    generate_subcommands.add_subcommand("full", generate_full, help="For models finetuned with `litgpt finetune full`.")
    generate_subcommands.add_subcommand("lora", generate_lora, help="For models finetuned with `litgpt finetune lora`.")
    generate_subcommands.add_subcommand("adapter", generate_adapter, help="For models finetuned with `litgpt finetune adapter`.")
    generate_subcommands.add_subcommand("adapter_v2", generate_adapter_v2, help="For models finetuned with `litgpt finetune adapter_v2`.")
    generate_subcommands.add_subcommand("sequentially", generate_sequentially, help="Generation script that partitions layers across devices to be run sequentially.")
    generate_subcommands.add_subcommand("tp", generate_tp, help="Generation script that uses tensor parallelism to run across devices.")

    convert_subcommands = convert.add_subcommands()
    convert_pretrained_checkpoint = ArgumentParser()
    convert_hf_checkpoint = ArgumentParser()
    convert_lit_checkpoint = ArgumentParser()
    from litgpt.scripts.convert_pretrained_checkpoint import convert_pretrained_checkpoint as convert_pretrained_checkpoint_fn
    from litgpt.scripts.convert_hf_checkpoint import convert_hf_checkpoint as convert_hf_checkpoint_fn
    from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint as convert_lit_checkpoint_fn
    convert_pretrained_checkpoint.add_function_arguments(convert_pretrained_checkpoint_fn)
    convert_hf_checkpoint.add_function_arguments(convert_hf_checkpoint_fn)
    convert_lit_checkpoint.add_function_arguments(convert_lit_checkpoint_fn)
    convert_subcommands.add_subcommand("to_litgpt", convert_hf_checkpoint, help="Convert Hugging Face weights to LitGPT weights.")
    convert_subcommands.add_subcommand("from_litgpt", convert_lit_checkpoint, help="Convert LitGPT weights to Hugging Face weights.")
    convert_subcommands.add_subcommand("pretrained_checkpoint", convert_pretrained_checkpoint, help="Convert a checkpoint after pretraining.")

    args = parser.parse_args()
    args_init = parser.instantiate_classes(args)
    print(args)
    print(args_init)


    subcommand = args_init.get("subcommand")
    subargs = args_init.get(subcommand)
    subsubcommand = subargs.get("subcommand")
    subsubargs = subargs.get(subsubcommand) if isinstance(subsubcommand, str) else None

    subcommands.add_subcommand("pretrain", pretrain, help="Pretrain a model.")
    subcommands.add_subcommand("finetune", finetune, help="Finetune a model with one of our existing methods.")
    subcommands.add_subcommand("generate", generate, help="Generate text samples based on a model and tokenizer.")
    subcommands.add_subcommand("chat", chat, help="Chat with a model.")
    subcommands.add_subcommand("convert", convert, help="Utilities to convert from and to LitGPT.")
    subcommands.add_subcommand("download", download, help="Download weights or tokenizer data from the Hugging Face Hub.")
    # TODO: should this be at the first level?
    subcommands.add_subcommand("merge_lora", merge_lora, help="Merges the LoRA weights with the base model.")

    subcommands_to_fn = {
        #"pretrain": {"fn": pretrain_fn, "help": "Pretrain a model."},
        #"finetune": {
        #    "lora": finetune_lora_fn
        #}
    }

    torch.set_float32_matmul_precision("high")

    level_1 = subcommands_to_fn[subcommand]
    if subsubcommand is None:
        level_1(**subargs)
    else:
        level_2 = level_1[subsubcommand]
        level_2(**subsubargs)


if __name__ == "__main__":
    main()
