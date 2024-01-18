# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json
import sys
from pathlib import Path
from typing import Optional

import requests
import torch
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer


def download_if_missing(file_path: Path, file_url: str):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists() and file_path.stat().st_size > 0:
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)


def load_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare(
    destination_path: Path = Path("data/flan"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    mask_inputs: bool = False,  # as in alpaca-lora
    subsets: Optional[str] = None,
    ignore_index: int = -1,
    max_seq_length: Optional[int] = None,
) -> None:
    """Prepare the FLAN-collection datasets for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.

    Since the original test set does not have responses, the validation set
    is used as the test set.
    """

    supported_subsets = {
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

    if subsets is not None:
        subsets = subsets.split(",")
        for sub in subsets:
            if sub not in supported_subsets:
                raise ValueError(f"{sub} not in {supported_subsets}")
    else:
        subsets = list(supported_subsets)

    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    print("Loading data file...")

    base_url = "https://huggingface.co/datasets/Muennighoff/flan/resolve/main/"

    train_set, test_set = [], []
    for sub in subsets:
        train_sub = sub + "_train"
        data_file_name = train_sub + ".jsonl"
        data_file_path = destination_path / data_file_name
        data_file_url = base_url + "train/" + data_file_name

        print(f"Loading training data file {sub}...")
        download_if_missing(data_file_path, data_file_url)
        sub_train_set = load_jsonl(data_file_path)
        train_set.extend(sub_train_set)

        test_sub = sub + "_test"
        data_file_name = test_sub + ".jsonl"
        data_file_path = destination_path / data_file_name
        data_file_url = base_url + "test/" + data_file_name

        print(f"Loading test data file {sub}...")
        download_if_missing(data_file_path, data_file_url)
        sub_test_set = load_jsonl(data_file_path)
        test_set.extend(sub_test_set)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool, ignore_index: int):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["targets"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {**example, "input_ids": encoded_full_prompt_and_response, "labels": labels}


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['inputs']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
