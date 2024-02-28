# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Set

import torch
from torch.utils.data import DataLoader
from lit_gpt.data import SFTDataset, get_sft_collate_fn, LitDataModule
from lit_gpt.data.alpaca import download_if_missing
from lit_gpt.tokenizer import Tokenizer

_URL = "https://huggingface.co/datasets/Muennighoff/flan/resolve/main"


class FLAN(LitDataModule):
    """FLAN data module for supervised finetuning.

    Provides train- and val-dataloaders. The batches return keys "input_ids" and "labels".
    """

    def __init__(
        self,
        mask_prompt: bool = False,
        test_split_fraction: float = 0.03865,  # to get exactly 2000 test samples,
        ignore_index: int = -1,
        seed: int = 42,
        num_workers: int = 4,
        data_url: str = _URL,
        download_dir: Path = Path("./data/flan"),
        subsets: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.mask_prompt = mask_prompt
        self.test_split_fraction = test_split_fraction
        self.ignore_index = ignore_index
        self.seed = seed
        self.num_workers = num_workers
        self.data_url = data_url
        self.download_dir = download_dir

        supported_subsets = _supported_subsets()
        if subsets is not None:
            self.subsets = subsets.split(",")
            for subset in self.subsets:
                if subset not in supported_subsets:
                    raise ValueError(f"{subset} not in {supported_subsets}")
        else:
            self.subsets = list(supported_subsets)

        self.tokenizer: Optional[Tokenizer] = None
        self.batch_size: int = 1
        self.max_seq_length: int = -1
        self.train_dataset: Optional[SFTDataset] = None
        self.test_dataset: Optional[SFTDataset] = None

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        for subset in self.subsets:
            for split in ("train", "test"):
                data_file_path = self.download_dir / f"{subset}_{split}.jsonl"
                data_file_url = f"{self.data_url}/{split}/{subset}_{split}.jsonl"
                download_if_missing(data_file_path, data_file_url)

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

    def _dataloader(self, split: str) -> DataLoader:
        data = []
        for subset in self.subsets:
            data_file_path = self.download_dir / f"{subset}_{split}.jsonl"
            data.extend(load_jsonl(data_file_path))

        for item in data:
            item["output"] = item.pop("targets")

        train_dataset = SFTDataset(
            data=data,
            tokenizer=self.tokenizer,
            prompt_template=prompt_template,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index)
        )


def load_jsonl(filename: Path) -> List[Dict[str, str]]:
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prompt_template(example: dict) -> str:
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['inputs']}\n\n### Response:\n"
    )


def _supported_subsets() -> Set[str]:
    return {
        "aeslc_10templates",
        "ag_news_subset_10templates",
        "anli_r1_10templates",
        "anli_r2_10templates",
        "anli_r3_10templates",
        "arc_challenge_10templates",
        "arc_easy_10templates",
        "bool_q_10templates",
        "cb_10templates",
        "cnn_dailymail_10templates",
        "cola_10templates",
        "common_gen_10templates",
        "copa_10templates",
        "coqa_10templates",
        "cosmos_qa_10templates",
        "dart_10templates",
        "definite_pronoun_resolution_10templates",
        "drop_10templates",
        "e2e_nlg_10templates",
        "fix_punct_10templates",
        "gigaword_10templates",
        "glue_mrpc_10templates",
        "glue_qqp_10templates",
        "hellaswag_10templates",
        "imdb_reviews_10templates",
        "math_dataset_10templates",
        "mnli_matched_10templates",
        "mnli_mismatched_10templates",
        "multi_news_10templates",
        "multirc_10templates",
        "natural_questions_10templates",
        "openbookqa_10templates",
        "opinion_abstracts_idebate_10templates",
        "opinion_abstracts_rotten_tomatoes_10templates",
        "para_crawl_enes_10templates",
        "paws_wiki_10templates",
        "piqa_10templates",
        "qnli_10templates",
        "quac_10templates",
        "record_10templates",
        "rte_10templates",
        "samsum_10templates",
        "sentiment140_10templates",
        "snli_10templates",
        "squad_v1_10templates",
        "squad_v2_10templates",
        "sst2_10templates",
        "story_cloze_10templates",
        "stsb_10templates",
        "trec_10templates",
        "trivia_qa_10templates",
        "true_case_10templates",
        "web_nlg_en_10templates",
        "wic_10templates",
        "wiki_lingua_english_en_10templates",
        "wmt14_enfr_10templates",
        "wmt16_translate_csen_10templates",
        "wmt16_translate_deen_10templates",
        "wmt16_translate_fien_10templates",
        "wmt16_translate_roen_10templates",
        "wmt16_translate_ruen_10templates",
        "wmt16_translate_tren_10templates",
        "wnli_10templates",
        "word_segment_10templates",
        "wsc_10templates",
        "yelp_polarity_reviews_10templates",
    }