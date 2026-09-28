# SFT architecture

## Current SFT flow

```text
raw example
  -> DataModule
  -> SFTDataset.__getitem__
  -> PromptStyle.apply(instruction)
  -> tokenize prompt and output separately
  -> concatenate prompt + output
  -> clone labels
  -> optionally mask prompt prefix
  -> collate and pad
  -> model(input_ids)
  -> logits[..., :-1] vs labels[..., 1:]
  -> chunked_cross_entropy(ignore_index=-100)
```

The normal representation is:

```python
{"instruction": "...", "output": "..."}
```

`SFTDataset` creates `input_ids` from the formatted instruction followed by the output. When `mask_prompt=True`, labels covering the encoded prompt prefix are replaced with `ignore_index`. When false, labels initially equal `input_ids`.

The collator right-pads `input_ids` with `pad_id` and labels with `ignore_index`.

Full and LoRA fine-tuning both consume the same `input_ids` and `labels`. They shift the causal objective with `logits[..., :-1, :]` and `labels[..., 1:]`. The existing loss implementation already ignores `ignore_index`.

## Current multi-turn handling

Deita and LIMA contain message lists in their source data, but their formatting functions currently produce separate instruction/output records for each user/assistant pair:

```text
messages
  -> question1/response1 example
  -> question2/response2 example
```

Thus earlier turns are not retained as context within one SFT example.

## Proposed controlled flow

```text
messages list
  -> one conversation-capable PromptStyle serialization
  -> tokenize complete serialized string once
  -> map assistant message spans to token positions
  -> labels = ignore_index everywhere except assistant content + assistant terminator
  -> existing collator
  -> existing full/LoRA training loops
  -> existing cross-entropy loss
```

Llama 3 is the preferred initial prompt style. Its `apply` method accepts message lists, emits role headers and `<|eot_id|>` terminators, and appends a final empty assistant generation header.

Expected assistant-only labels:

```text
system header/content/eot       ignored
user header/content/eot         ignored
assistant header                ignored
assistant content               trainable
assistant eot                   trainable
```

This applies to every assistant message, not only the final one. BOS and role/control tokens remain in `input_ids` but are not trainable targets.

## Components expected to remain unchanged

- `litgpt/finetune/full.py` training and validation loops
- `litgpt/finetune/lora.py` training and validation loops
- `litgpt/utils.py:chunked_cross_entropy`
- the existing instruction/output data path
- the existing padding behavior

The principal implementation boundary is the SFT data/label-construction path. Exact token-span mapping must be validated before public API decisions are made.

The initial implementation exposes the optional strategy through the generic JSON data module and supports assistant masking only with Llama 3 plus a Hugging Face tokenizer that supplies token offsets.

Real Llama 3 tokenizer validation showed that the backend may emit an extra leading BOS ID with offset `(0, 0)`; the data path aligns normalized LitGPT IDs to backend IDs before applying offsets, and truncates offsets with the retained sequence. Llama 3 role headers, whitespace tokens, and `<|eot_id|>` receive ordinary character spans; assistant content and assistant `<|eot_id|>` are selected successfully.
