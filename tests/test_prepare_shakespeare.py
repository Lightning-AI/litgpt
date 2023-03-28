import os
import subprocess
import sys
from pathlib import Path

wd = (Path(__file__).parent.parent / "scripts").absolute()


def test_prepare(tmp_path):
    sys.path.append(str(wd))

    import prepare_shakespeare

    prepare_shakespeare.prepare(tmp_path)

    assert set(os.listdir(tmp_path)) == {"train.bin", "tokenizer.model", "tokenizer.vocab", "input.txt", "val.bin"}


def test_cli():
    cli_path = wd / "prepare_shakespeare.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert 'Prepare the "Tiny Shakespeare"' in output
