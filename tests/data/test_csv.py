# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from unittest import mock


@mock.patch("lit_gpt.data.csv.prompt_template", "X: {instruction} {input} Y:")
def test_csv(tmp_path, mock_tockenizer):
    from lit_gpt.data import CSV

    csv_path = tmp_path / "data.csv"
    mock_data = (
        "instruction,input,output\n"
        "Add,2+2,4\n"
        "Subtract,5-3,2\n"
        "Multiply,6*4,24\n"
        "Divide,10/2,5\n"
        "Exponentiate,2^3,8\n"
        "Square root,√9,3\n"
    )
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write(mock_data)

    # TODO: Make prompt template an argumenet
    csv = CSV(csv_path, test_split_fraction=0.5, num_workers=0)
    csv.connect(tokenizer=mock_tockenizer, batch_size=2)
    csv.prepare_data()  # does nothing
    csv.setup()

    train_dataloader = csv.train_dataloader()
    val_dataloader = csv.val_dataloader()

    assert len(train_dataloader) == 2
    assert len(val_dataloader) == 2

    train_data = list(train_dataloader)
    val_data = list(val_dataloader)

    assert train_data[0]["input_ids"].size(0) == 2
    assert train_data[1]["input_ids"].size(0) == 1
    assert val_data[0]["input_ids"].size(0) == 2
    assert val_data[1]["input_ids"].size(0) == 1

    assert mock_tockenizer.decode(train_data[0]["input_ids"][0]).startswith("X: Divide 10/2 Y:5")
    assert mock_tockenizer.decode(train_data[0]["input_ids"][1]).startswith("X: Add 2+2 Y:4")
    assert mock_tockenizer.decode(train_data[1]["input_ids"][0]).startswith("X: Multiply 6*4 Y:24")

    assert mock_tockenizer.decode(val_data[0]["input_ids"][0]).startswith("X: Exponentiate 2^3 Y:8")
    assert mock_tockenizer.decode(val_data[0]["input_ids"][1]).startswith("X: Subtract 5-3 Y:2")
    assert mock_tockenizer.decode(val_data[1]["input_ids"][0]).startswith("X: Square root √9 Y:3")
