import sys
from pathlib import Path
from typing import List, Optional

from litgpt.utils import CLI


def parser_commands() -> List[str]:
    return [
        "download",
        "chat",
        "finetune",
        "finetune_lora",
        "finetune_full",
        "finetune_adapter",
        "finetune_adapter_v2",
        "pretrain",
        "generate",
        "generate_full",
        "generate_adapter",
        "generate_adapter_v2",
        "generate_sequentially",
        "generate_speculatively",
        "generate_tp",
        "convert_to_litgpt",
        "convert_from_litgpt",
        "convert_pretrained_checkpoint",
        "merge_lora",
        "evaluate",
        "serve",
    ]


def save_hyperparameters(
    function: callable,
    checkpoint_dir: Path,
    known_commands: Optional[List[str]] = None,
) -> None:
    """Captures the CLI parameters passed to `function` without running `function` and saves them to the checkpoint."""
    from jsonargparse import capture_parser

    # TODO: Make this more robust
    # This hack strips away the subcommands from the top-level CLI
    # to parse the file as if it was called as a script
    if known_commands is None:
        known_commands = parser_commands()
    known_commands = [(c,) for c in known_commands]
    for known_command in known_commands:
        unwanted = slice(1, 1 + len(known_command))
        if tuple(sys.argv[unwanted]) == known_command:
            sys.argv[unwanted] = []

    parser = capture_parser(lambda: CLI(function))
    config = parser.parse_args()
    parser.save(config, checkpoint_dir / "hyperparameters.yaml", overwrite=True)
