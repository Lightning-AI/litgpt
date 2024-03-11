# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from typing import Any, Literal

import torch


def pretrain(*args: Any, **kwargs: Any) -> Any:
    """Pretrain a model."""
    from litgpt.pretrain import setup

    return setup(*args, **kwargs)


def finetune(method: Literal["full", "lora", "qlora", "adapter", "adapter_v2"], *args: Any, **kwargs: Any) -> Any:
    """Finetune a model with one of our existing methods."""
    if method in ("lora", "qlora"):
        from litgpt.finetune.lora import setup

        if method == "qlora":
            kwargs.setdefault("quantize", "bnb.nf4-dq")
            kwargs.setdefault("precision", "bf16-true")
        return setup(*args, **kwargs)
    # FIXME
    # elif method == "full":
    #    from litgpt.finetune.full import setup
    #    return setup(*args, **kwargs)
    # elif method == "adapter":
    #    from litgpt.finetune.adapter import setup
    #    return setup(*args, **kwargs)
    # elif method == "adapter_v2":
    #    from litgpt.finetune.adapter_v2 import setup
    #    return setup(*args, **kwargs)
    # note: do not call `setup` outside of the `if` blocks because jsonargparse doesn't understand it
    raise NotImplementedError(method)


def chat(*args: Any, **kwargs: Any) -> Any:
    """Chat with a model."""
    from litgpt.chat.base import main

    return main(*args, **kwargs)


def generate(
    method: Literal["base", "full", "lora", "adapter", "adapter_v2", "sequentially", "tp"], *args: Any, **kwargs: Any
) -> Any:
    """Generate text samples based on a model and tokenizer."""
    if method == "base":
        from litgpt.generate.base import main

        return main(*args, **kwargs)
    elif method == "full":
        from litgpt.generate.full import main

        return main(*args, **kwargs)
    elif method == "lora":
        from litgpt.generate.lora import main

        return main(*args, **kwargs)
    elif method == "adapter":
        from litgpt.generate.adapter import main

        return main(*args, **kwargs)
    elif method == "adapter_v2":
        from litgpt.generate.adapter_v2 import main

        return main(*args, **kwargs)
    elif method == "sequentially":
        from litgpt.generate.sequentially import main

        return main(*args, **kwargs)
    elif method == "tp":
        from litgpt.generate.tp import main

        return main(*args, **kwargs)
    raise NotImplementedError(method)


def main():
    from litgpt.utils import CLI

    torch.set_float32_matmul_precision("high")

    from litgpt.scripts.convert_pretrained_checkpoint import convert_pretrained_checkpoint
    from litgpt.scripts.convert_hf_checkpoint import convert_hf_checkpoint
    from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
    from litgpt.scripts.download import download
    from litgpt.scripts.merge_lora import merge_lora

    CLI(
        [
            pretrain,
            finetune,
            chat,
            generate,
            convert_hf_checkpoint,
            convert_lit_checkpoint,
            convert_pretrained_checkpoint,
            download,
            merge_lora,
        ]
    )


if __name__ == "__main__":
    main()
