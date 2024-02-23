from typing import List, Dict, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from lit_gpt import Tokenizer


class SFTDataset(Dataset):
    """A dataset for supervised finetuning with `input_ids` and `labels`."""
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Tokenizer,
        prompt_template: Union[str, callable],
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -1,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.mask_prompt = mask_prompt
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Processes a single sample.

        Returns a dict with two keys:
            input_ids: The encoded prompt + response
            labels: Same as input_ids, unless ``mask_prompt=True`` in which case the 'prompt' part is replaced with
                the ``ignore_index``.
        """
        example = self.data[idx]

        if isinstance(self.prompt_template, str):
            prompt = self.prompt_template.format(instruction=example["instruction"], input=example["input"])
        else:
            prompt = self.prompt_template(instruction=example["instruction"], input=example["input"])

        prompt_and_response = prompt + example["output"]
        encoded_prompt = self.tokenizer.encode(prompt, max_length=self.max_seq_length)
        encoded_prompt_and_response = self.tokenizer.encode(
            prompt_and_response, eos=True, max_length=self.max_seq_length
        )

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_prompt_and_response.clone()
        if self.mask_prompt:
            labels[: len(encoded_prompt)] = self.ignore_index

        return {"input_ids": encoded_prompt_and_response, "labels": labels}


def sft_collate_fn(samples: List[Dict[str, Tensor]], max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -1) -> Dict[str, Tensor]:
    longest = max(len(sample["input_ids"]) for sample in samples)
    max_length = max_seq_length if max_seq_length > 0 else longest

    batched = {}
    for key in ("input_ids", "labels"):
        pad_value = pad_id if key == "input_ids" else ignore_index
        batched[key] = torch.stack([
            torch.nn.functional.pad(sample[key], (0, longest - len(sample[key])), value=pad_value)
            for sample in samples
        ])
        batched[key] = batched[key][:, :max_length]
    return batched
