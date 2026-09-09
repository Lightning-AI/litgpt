import sys
from pathlib import Path

from litgpt.utils import CLI


def parser_commands() -> list[str]:
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
        "validate",
    ]


def save_hyperparameters(
    function: callable,
    checkpoint_dir: Path,
    known_commands: list[str] | None = None,
) -> None:
    """Captures the CLI parameters passed to `function` without running `function` and saves them to the checkpoint."""
    from jsonargparse import capture_parser

    if known_commands is None:
        known_commands = parser_commands()
    args = sys.argv[1:]
    known_commands = [(c,) for c in known_commands]
    known_commands.extend(
        [
            ("finetune", "full"),
            ("finetune", "lora"),
            ("finetune", "adapter"),
            ("finetune", "adapter_v2"),
        ]
    )
    found_known = False
    for known_command in sorted(known_commands, key=len, reverse=True):
        if tuple(args[: len(known_command)]) == known_command:
            args = args[len(known_command) :]
            found_known = True
            break

    parser = capture_parser(lambda: CLI(function))
    try:
        config = parser.parse_args(args)
    except SystemExit:
        if not found_known:
            return
        raise
    parser.save(config, checkpoint_dir / "hyperparameters.yaml", overwrite=True)
