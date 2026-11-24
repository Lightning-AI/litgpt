# Relevant files

## SFT/data pipeline

- `litgpt/data/base.py`
  - `DataModule`: base data-module interface.
  - `SFTDataset`: tokenizes instruction/output examples and constructs labels.
  - `SFTDataset.__getitem__`: prompt formatting, tokenization, concatenation, truncation, and prompt masking.
  - `get_sft_collate_fn` / `_sft_collate_fn`: batch padding and batch truncation.
- `litgpt/data/alpaca.py`
  - Standard instruction/output SFT data module; passes `mask_prompt` and `ignore_index` to `SFTDataset`.
- `litgpt/data/json_data.py`
  - Generic JSON/JSONL SFT data module using the same dataset path; now forwards the optional masking strategy.

## Prompt formatting

- `litgpt/prompts.py`
  - `PromptStyle`: prompt-formatting interface.
  - `Llama3.apply`: serializes string or message-list prompts with role headers, `<|eot_id|>`, and a final assistant header.
  - `R1Base.apply`: another message-aware style, not part of the initial scope.
  - `save_prompt_style` / `load_prompt_style`: checkpoint prompt-style persistence; relevant for understanding compatibility.
- `tests/test_prompts.py`
  - `test_multiturn_prompt`: verifies Llama 3 multi-turn serialization.

## Tokenization

- `litgpt/tokenizer.py`
  - `Tokenizer`: supports Hugging Face `tokenizer.json` and SentencePiece backends.
  - `Tokenizer.encode`: backend tokenization, BOS/EOS handling, and truncation.
  - The wrapper currently returns token IDs and does not expose token offsets; assistant masking accesses validated Hugging Face backend offsets directly.

## Training/loss

- `litgpt/finetune/full.py`
  - `fit`: full-parameter SFT training and causal label shifting.
  - `validate`: masked validation loss.
- `litgpt/finetune/lora.py`
  - `fit`: LoRA SFT training using the same labels and causal shift.
  - `validate`: masked validation loss.
- `litgpt/utils.py`
  - `chunked_cross_entropy`: cross-entropy with `ignore_index` support and optional chunking.

## Existing conversation datasets

- `litgpt/data/deita.py`
  - `format_dataset`: currently splits source message lists into independent instruction/output examples.
  - `Deita`: data-module configuration, including `include_multiturn_conversations`.
- `litgpt/data/lima.py`
  - `format_dataset`: similarly converts conversation turns into separate instruction/output examples.

## Tests and fixtures

- `tests/data/test_base.py`
  - `test_sft_dataset`: prompt masking, unmasked labels, EOS, custom ignore index, and truncation.
  - `test_sft_collate_fn_padding`: padding labels with `ignore_index`.
  - `test_sft_collate_fn_truncation`: batch truncation.
- `tests/data/test_deita.py`
  - `test_format_dataset`: verifies current Deita conversation splitting.
  - `test_deita`: verifies construction of `SFTDataset` data loaders.
- `tests/conftest.py`
  - `MockTokenizer`: deterministic character-level tokenizer useful for label-mask tests.
