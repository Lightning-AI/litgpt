import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, call, ANY

wd = (Path(__file__).parent.parent / "scripts").absolute()

import requests


def maybe_get_file(url, file_path):
    if not file_path.exists():
        with open(file_path, "w") as f:
            f.write(requests.get(url).text)


def test_prepare_sample(tmp_path):
    sys.path.append(str(wd))

    tmp_path.mkdir(parents=True, exist_ok=True)

    vocabulary_path = tmp_path / "tokenizer.json"
    maybe_get_file("https://huggingface.co/stabilityai/stablelm-base-alpha-3b/raw/main/tokenizer.json", vocabulary_path)

    tokenizer_path = tmp_path / "tokenizer_config.json"
    maybe_get_file("https://huggingface.co/stabilityai/stablelm-base-alpha-3b/raw/main/tokenizer_config.json", tokenizer_path)

    sample_path = tmp_path / "sample"
    source_path = sample_path / "source"
    dest_path = sample_path / "dest"

    source_path.mkdir(parents=True, exist_ok=True)

    sample = {
        "meta": {"some": "info"},
        "text": "some text"
    }

    jsonl_sample = "\n".join([json.dumps(el) for el in [sample] * 2])

    import prepare_redpajama

    for filename in prepare_redpajama.filenames_sample:
        with open(source_path / filename, "w") as f:
            f.write(jsonl_sample)

    prepare_redpajama.prepare(
        source_path=source_path,
        vocabulary_path=vocabulary_path,
        tokenizer_path=tokenizer_path,
        destination_path=dest_path,
        sample=True
    )

    bin_files = [el.replace(".jsonl", "_0000000000.bin") for el in prepare_redpajama.filenames_sample]

    assert set(os.listdir(dest_path)) == set(bin_files)

    from lit_stablelm import Tokenizer
    from lit_stablelm.packed_dataset import PackedDataset

    tokenizer = Tokenizer(
        vocabulary_path=vocabulary_path,
        config_path=tokenizer_path
    )

    # artificially set block_size to fit the text
    block_size = len(tokenizer.encode("some text"))

    for filename in bin_files:
        filenames = [os.path.join(dest_path, filename)]
        dataset = PackedDataset(filenames=filenames, n_chunks=1, block_size=block_size, shuffle=False)
        dataset_iter = iter(dataset)
        assert tokenizer.decode(next(dataset_iter)) == "some text"
        assert tokenizer.decode(next(dataset_iter)) == "some text"


def test_prepare_full(tmp_path):
    sys.path.append(str(wd))

    tmp_path.mkdir(parents=True, exist_ok=True)

    vocabulary_path = tmp_path / "tokenizer.json"
    maybe_get_file("https://huggingface.co/stabilityai/stablelm-base-alpha-3b/raw/main/tokenizer.json", vocabulary_path)

    tokenizer_path = tmp_path / "tokenizer_config.json"
    maybe_get_file("https://huggingface.co/stabilityai/stablelm-base-alpha-3b/raw/main/tokenizer_config.json", tokenizer_path)

    full_path = tmp_path / "full"
    source_path = full_path / "source"
    dest_path = full_path / "dest"

    source_path.mkdir(parents=True, exist_ok=True)

    sample = {
        "meta": {"some": "info"},
        "text": "some text"
    }

    jsonl_sample = "\n".join([json.dumps(el) for el in [sample] * 2])

    import prepare_redpajama

    arxiv_file = source_path / "arxiv" / "arxiv_0.jsonl"
    arxiv_file.parent.mkdir(parents=True, exist_ok=True)
    with open(arxiv_file, "w") as f:
        f.write(jsonl_sample)

    import zstandard as zstd

    cc_file = source_path / "common_crawl" / "cc_0.jsonl"
    cc_file.parent.mkdir(parents=True, exist_ok=True)
    with zstd.open(cc_file, "wt", encoding="utf-8") as f:
        f.write(jsonl_sample)

    filename_sets = {
        "arxiv": "arxiv/arxiv*",
        "common_crawl": "common_crawl/*",
    }

    with mock.patch("prepare_redpajama.filename_sets", filename_sets):
        prepare_redpajama.prepare(
            source_path=source_path,
            vocabulary_path=vocabulary_path,
            tokenizer_path=tokenizer_path,
            destination_path=dest_path,
            sample=False
        )

        all_names = prepare_redpajama.filename_sets.keys()
        bin_files = [el + "_0000000000.bin" for el in all_names]

    assert set(os.listdir(dest_path)) == set(bin_files)

    from lit_stablelm import Tokenizer
    from lit_stablelm.packed_dataset import PackedDataset

    tokenizer = Tokenizer(
        vocabulary_path=vocabulary_path,
        config_path=tokenizer_path
    )

    # artificially set block_size to fit the text
    block_size = len(tokenizer.encode("some text"))

    filenames = [os.path.join(dest_path, el) for el in bin_files]

    for filename in filenames:
        dataset = PackedDataset(filenames=[filename], n_chunks=1, block_size=block_size, shuffle=False)
        dataset_iter = iter(dataset)
        assert tokenizer.decode(next(dataset_iter)) == "some text"
        assert tokenizer.decode(next(dataset_iter)) == "some text"


def test_cli():
    cli_path = wd / "prepare_redpajama.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert 'Prepare the "Red Pajama"' in output
