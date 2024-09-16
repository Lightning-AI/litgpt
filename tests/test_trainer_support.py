# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import sys
from pathlib import Path
import pytest
from tests.conftest import RunIf

import torch
import litgpt
from litgpt.api import LLM
from litgpt.data import Alpaca2k
import lightning as L

