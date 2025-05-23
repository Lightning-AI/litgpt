import json

import torch
from litdata import TokensLoader, optimize
from torch.utils._pytree import tree_map

from litgpt.data.text_files import TextFiles


class Tokenizer:
    bos_id = 0

    def encode(self, text, bos, eos):
        assert bos
        assert not eos
        return [self.bos_id] + [ord(c) for c in text]


def tokenize(data):
    for story in data:
        yield torch.tensor(story)


def fake_chunk(path, data):
    optimize(
        fn=tokenize,
        inputs=[data] * len(data),
        output_dir=str(path),
        num_workers=1,
        chunk_bytes="200MB",
        item_loader=TokensLoader(),
    )


def test_textfiles_datamodule(tmp_path):
    from litgpt.data.text_files import TextFiles

    data_dir = tmp_path / "textfiles"
    datamodule = TextFiles(train_data_path=data_dir, num_workers=1)
    datamodule.connect(max_seq_length=2, tokenizer=Tokenizer())

    # simulate `datamodule.prepare_data`
    train_data_dir = data_dir / "train"
    train_data_dir.mkdir(parents=True)
    fake_chunk(train_data_dir, [[12], [0, 23, 15, 63, 0], [73, 5, 0, 1, 1999, 0, 13]])
    datamodule.setup()

    tr_dataloader = datamodule.train_dataloader()
    tr_dataloader.shuffle = False

    actual = tree_map(torch.Tensor.tolist, list(tr_dataloader))

    # there is 1 sample per index in the data (13)
    assert actual == [
        [[73, 5, 0]],
        [[12, 0, 23]],
        [[5, 0, 1]],
        [[0, 73, 5]],
        [[1999, 0, 13]],
        [[0, 1, 1999]],
        [[1, 1999, 0]],
        [[0, 23, 15]],
        [[13, 12, 0]],
        [[63, 0, 73]],
        [[23, 15, 63]],
        [[15, 63, 0]],
        [[0, 13, 12]],
    ]


class MockTokenizer:
    bos_id = 0
    eos_id = 1
    use_bos = True

    def encode(self, text, bos=True, eos=False, device=None, max_length=-1):
        # Simple: map each character to its ordinal + 2
        tokens = [ord(c) + 2 for c in text]
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens.append(self.eos_id)
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.long, device=device)

    def decode(self, tensor):
        ids = tensor.tolist() if tensor.ndim > 0 else [tensor.item()]
        chars = []
        for tid in ids:
            if tid == self.bos_id:
                chars.append("<BOS>")
            elif tid == self.eos_id:
                chars.append("<EOS>")
            else:
                chars.append(chr(tid - 2))
        return "".join(chars)

    def decode_stream(self, token_stream, device=None):
        for token in token_stream:
            yield self.decode(token)

    @property
    def vocab_size(self):
        return 130


def test_textfiles_token_loader(tmp_path):
    # Create the directory for text files
    data_dir = tmp_path / "textfiles"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Write sample training data to the directory
    sample_texts = ["hello world", "foo bar", "lorem ipsum"]
    for i, text in enumerate(sample_texts):
        (data_dir / f"{i}.txt").write_text(text)

    datamodule = TextFiles(train_data_path=data_dir, num_workers=1)
    datamodule.connect(max_seq_length=2, tokenizer=MockTokenizer())
    datamodule.prepare_data()

    # ensure training set uses tokens loader
    index_json = data_dir / "train" / "index.json"
    assert index_json.exists()
    meta = json.loads(index_json.read_text())
    assert meta["config"]["item_loader"] == "TokensLoader"

    # ensure validation set uses tokens loader
    index_json = data_dir / "val" / "index.json"
    assert index_json.exists()
    meta = json.loads(index_json.read_text())
    assert meta["config"]["item_loader"] == "TokensLoader"
