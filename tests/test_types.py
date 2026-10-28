# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from typing import get_args

from litgpt.constants import _SUPPORTED_LOGGERS
from litgpt.types import LoggerChoice


def test_logger_types_match_constants():
    """Ensure LoggerChoice and _SUPPORTED_LOGGERS stay synchronized."""
    logger_choice_args = get_args(LoggerChoice)
    assert logger_choice_args == _SUPPORTED_LOGGERS, (
        f"LoggerChoice type args {logger_choice_args} != "
        f"_SUPPORTED_LOGGERS {_SUPPORTED_LOGGERS}. "
        f"These must stay synchronized. Update both litgpt/types.py and "
        f"litgpt/constants.py when adding new loggers."
    )
