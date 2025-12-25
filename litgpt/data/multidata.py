# # Copyright Lightning AI. Licensed under the Apache License 2.0
# """Unified SFT DataModule for Medical-O1 and Indic-Instruct"""

# import os
# from dataclasses import dataclass, field
# from typing import List, Dict, Optional, Union, Literal
# from venv import logger

# import torch
# from torch.utils.data import DataLoader, random_split

# from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
# from litgpt.prompts import PromptStyle
# from litgpt.tokenizer import Tokenizer


# # ---------------------------------------------------------------------
# # Dataset formatters
# # ---------------------------------------------------------------------

# def format_medical_o1(dataset_partition: List[Dict]) -> List[Dict]:
#     """Question + <unused0> CoT <unused1> + Final Answer"""
#     formatted = []

#     for entry in dataset_partition:
#         output = (
#             "<unused0>"
#             f"{entry['Complex_CoT'].strip()}"
#             "<unused1>\n\n"
#             f"{entry['Response'].strip()}"
#         )

#         formatted.append({
#             "instruction": entry["Question"].strip(),
#             "input": "",
#             "output": output,
#         })

#     return formatted


# def format_indic_instruct(
#     dataset_partition: List[Dict],
#     include_multiturn: bool,
# ) -> List[Dict]:
#     formatted = []

#     for entry in dataset_partition:
#         convo = entry["messages"]

#         if include_multiturn:
#             for i in range(0, len(convo) - 1, 2):
#                 formatted.append({
#                     "instruction": convo[i]["content"],
#                     "input": "",
#                     "output": convo[i + 1]["content"],
#                 })
#         else:
#             formatted.append({
#                 "instruction": convo[0]["content"],
#                 "input": "",
#                 "output": convo[1]["content"],
#             })

#     return formatted


# def data_collator(self, batch):
#     logger.debug(f"Collating batch of size {len(batch)}")


# # ---------------------------------------------------------------------
# # Unified DataModule
# # ---------------------------------------------------------------------

# @dataclass
# class UnifiedSFTDataModule(DataModule):
#     """
#     Unified DataModule with dataset switch.

#     dataset_type:
#         - "medical_o1"
#         - "indic_instruct"
#     """

#     # ---- Switch ----
#     dataset_type: Literal["medical_o1", "indic_instruct"] = "medical_o1"

#     # ---- Common training params ----
#     mask_prompt: bool = False
#     val_split_fraction: float = 0.1
#     prompt_style: Union[str, PromptStyle] = "pragna-1b"
#     ignore_index: int = -100
#     seed: int = 42
#     num_workers: int = 4
#     include_multiturn_conversations: bool = True

#     # ---- Dataset repos ----
#     medical_repo_id: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
#     indic_repo_id: str = "ai4bharat/indic-instruct-data-v0.1"

#     access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))

#     # ---- Runtime ----
#     tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
#     batch_size: int = field(default=1, init=False, repr=False)
#     max_seq_length: int = field(default=-1, init=False, repr=False)

#     train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
#     test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

#     # -----------------------------------------------------------------

#     def __post_init__(self):
#         super().__init__()
#         if isinstance(self.prompt_style, str):
#             self.prompt_style = PromptStyle.from_name(self.prompt_style)

#         if self.dataset_type not in {"medical_o1", "indic_instruct"}:
#             raise ValueError(f"Invalid dataset_type: {self.dataset_type}")

#     def connect(
#         self,
#         tokenizer: Optional[Tokenizer] = None,
#         batch_size: int = 1,
#         max_seq_length: Optional[int] = None,
#     ) -> None:
#         self.tokenizer = tokenizer
#         self.batch_size = batch_size
#         self.max_seq_length = -1 if max_seq_length is None else max_seq_length

#     # -----------------------------------------------------------------

#     def prepare_data(self) -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             load_dataset(self.medical_repo_id, "en", token=self.access_token)
#         else:
#             load_dataset(self.indic_repo_id, "anudesh", token=self.access_token)

#     # -----------------------------------------------------------------

#     def setup(self, stage: str = "") -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             dataset = load_dataset(self.medical_repo_id, "en", token=self.access_token)
#             data = format_medical_o1(dataset["train"])

#         else:
#             dataset = load_dataset(self.indic_repo_id, "anudesh", token=self.access_token)
#             data = format_indic_instruct(
#                 dataset["hi"],
#                 self.include_multiturn_conversations,
#             )

#         # ---- Train / Val split ----
#         train_data, val_data = random_split(
#             data,
#             [1.0 - self.val_split_fraction, self.val_split_fraction],
#             generator=torch.Generator().manual_seed(self.seed),
#         )

#         self.train_dataset = SFTDataset(
#             data=list(train_data),
#             tokenizer=self.tokenizer,
#             prompt_style=self.prompt_style,
#             max_seq_length=self.max_seq_length,
#             mask_prompt=self.mask_prompt,
#             ignore_index=self.ignore_index,
#         )

#         self.test_dataset = SFTDataset(
#             data=list(val_data),
#             tokenizer=self.tokenizer,
#             prompt_style=self.prompt_style,
#             max_seq_length=self.max_seq_length,
#             mask_prompt=self.mask_prompt,
#             ignore_index=self.ignore_index,
#         )

#     # -----------------------------------------------------------------

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             generator=torch.Generator().manual_seed(self.seed),
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )




############################## three dataset ###################



# """Unified SFT DataModule for Medical-O1, Indic-Instruct, and KissanAI"""

# import os
# from dataclasses import dataclass, field
# from typing import List, Dict, Optional, Union, Literal
# from venv import logger

# import torch
# from torch.utils.data import DataLoader, random_split

# from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
# from litgpt.prompts import PromptStyle
# from litgpt.tokenizer import Tokenizer




# def format_medical_o1(dataset_partition: List[Dict]) -> List[Dict]:
#     """Question + <unused0> CoT <unused1> + Final Answer"""
#     formatted = []

#     for entry in dataset_partition:
#         output = (
#             "<unused0>"
#             f"{entry['Complex_CoT'].strip()}"
#             "<unused1>\n\n"
#             f"{entry['Response'].strip()}"
#         )

#         formatted.append({
#             "instruction": entry["Question"].strip(),
#             "input": "",
#             "output": output,
#         })

#     return formatted


# def format_indic_instruct(
#     dataset_partition: List[Dict],
#     include_multiturn: bool,
# ) -> List[Dict]:
#     formatted = []

#     for entry in dataset_partition:
#         convo = entry["messages"]

#         if include_multiturn:
#             for i in range(0, len(convo) - 1, 2):
#                 formatted.append({
#                     "instruction": convo[i]["content"],
#                     "input": "",
#                     "output": convo[i + 1]["content"],
#                 })
#         else:
#             formatted.append({
#                 "instruction": convo[0]["content"],
#                 "input": "",
#                 "output": convo[1]["content"],
#             })

#     return formatted


# def format_kissanai(
#     dataset_partition: List[Dict],
#     include_multiturn: bool,
# ) -> List[Dict]:
#     formatted = []

#     for entry in dataset_partition:
#         convo = entry["conversations"]

#         user_turns = [x["value"] for x in convo if x.get("from") == "user"]
#         assistant_turns = [x["value"] for x in convo if x.get("from") == "assistant"]

#         if not user_turns or not assistant_turns:
#             continue

#         def replace_think_tokens(text: str) -> str:
#             return (
#                 text
#                 .replace("<think>", "<unused0>")
#                 .replace("</think>", "<unused1>")
#             )

#         if include_multiturn:
#             for u, a in zip(user_turns, assistant_turns):
#                 formatted.append({
#                     "instruction": u.strip(),
#                     "input": "",
#                     "output": replace_think_tokens(a.strip()),
#                 })
#         else:
#             formatted.append({
#                 "instruction": user_turns[0].strip(),
#                 "input": "",
#                 "output": replace_think_tokens(assistant_turns[0].strip()),
#             })

#     if formatted:
#         logger.info(f"formatted_ds[0]: {formatted[0]}")

#     return formatted


# def data_collator(self, batch):
#     logger.debug(f"Collating batch of size {len(batch)}")



# @dataclass
# class UnifiedSFTDataModule(DataModule):
#     """
#     Unified SFT DataModule.

#     dataset_type:
#       - medical_o1
#       - indic_instruct
#       - kissanai
#     """

#     dataset_type: Literal[
#         "medical_o1",
#         "indic_instruct",
#         "kissanai",
#     ] = "medical_o1"

#     mask_prompt: bool = False
#     val_split_fraction: float = 0.1
#     prompt_style: Union[str, PromptStyle] = "pragna-1b"
#     ignore_index: int = -100
#     seed: int = 42
#     num_workers: int = 4
#     include_multiturn_conversations: bool = True


#     medical_repo_id: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
#     indic_repo_id: str = "ai4bharat/indic-instruct-data-v0.1"
#     kissanai_repo_id: str = "KissanAI/Thinking-climate-100k"

#     access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))


#     tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
#     batch_size: int = field(default=1, init=False, repr=False)
#     max_seq_length: int = field(default=-1, init=False, repr=False)

#     train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
#     test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)



#     def __post_init__(self):
#         super().__init__()
#         if isinstance(self.prompt_style, str):
#             self.prompt_style = PromptStyle.from_name(self.prompt_style)

#         if self.dataset_type not in {
#             "medical_o1",
#             "indic_instruct",
#             "kissanai",
#         }:
#             raise ValueError(f"Invalid dataset_type: {self.dataset_type}")

#     def connect(
#         self,
#         tokenizer: Optional[Tokenizer] = None,
#         batch_size: int = 2,
#         max_seq_length: Optional[int] = None,
#     ) -> None:
#         self.tokenizer = tokenizer
#         self.batch_size = batch_size
#         self.max_seq_length = -1 if max_seq_length is None else max_seq_length


#     def prepare_data(self) -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             load_dataset(self.medical_repo_id, "en", token=self.access_token)

#         elif self.dataset_type == "indic_instruct":
#             load_dataset(self.indic_repo_id, "anudesh", token=self.access_token)

#         else:
#             load_dataset(self.kissanai_repo_id, token=self.access_token)



#     def setup(self, stage: str = "") -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             ds = load_dataset(self.medical_repo_id, "en", token=self.access_token)
#             data = format_medical_o1(ds["train"])

#         elif self.dataset_type == "indic_instruct":
#             ds = load_dataset(self.indic_repo_id, "anudesh", token=self.access_token)
#             data = format_indic_instruct(
#                 ds["hi"],
#                 self.include_multiturn_conversations,
#             )

#         else:
#             ds = load_dataset(self.kissanai_repo_id, token=self.access_token)
#             data = format_kissanai(
#                 ds["train"],
#                 self.include_multiturn_conversations,
#             )

#         train_data, val_data = random_split(
#             data,
#             [1.0 - self.val_split_fraction, self.val_split_fraction],
#             generator=torch.Generator().manual_seed(self.seed),
#         )

#         self.train_dataset = SFTDataset(
#             data=list(train_data),
#             tokenizer=self.tokenizer,
#             prompt_style=self.prompt_style,
#             max_seq_length=self.max_seq_length,
#             mask_prompt=self.mask_prompt,
#             ignore_index=self.ignore_index,
#         )

#         self.test_dataset = SFTDataset(
#             data=list(val_data),
#             tokenizer=self.tokenizer,
#             prompt_style=self.prompt_style,
#             max_seq_length=self.max_seq_length,
#             mask_prompt=self.mask_prompt,
#             ignore_index=self.ignore_index,
#         )


#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             generator=torch.Generator().manual_seed(self.seed),
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )



# ############################## two dataset  with <unused0> and <unused1>###################

"""Unified SFT DataModule for Medical-O1 and KissanAI"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Literal
from venv import logger

import torch
from torch.utils.data import DataLoader, random_split

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer


# ---------------------------------------------------------------------
# Dataset formatters
# ---------------------------------------------------------------------

def format_medical_o1(dataset_partition: List[Dict]) -> List[Dict]:
    """Question + <unused0> CoT <unused1> + Final Answer"""
    formatted = []

    for entry in dataset_partition:
        output = (
            "<unused0>"
            f"{entry['Complex_CoT'].strip()}"
            "<unused1>\n\n"
            f"{entry['Response'].strip()}"
        )

        formatted.append({
            "instruction": entry["Question"].strip(),
            "input": "",
            "output": output,
        })

    return formatted


def format_kissanai(
    dataset_partition: List[Dict],
    include_multiturn: bool,
) -> List[Dict]:
    formatted = []

    for entry in dataset_partition:
        convo = entry["conversations"]

        user_turns = [x["value"] for x in convo if x.get("from") == "user"]
        assistant_turns = [x["value"] for x in convo if x.get("from") == "assistant"]

        if not user_turns or not assistant_turns:
            continue

        def replace_think_tokens(text: str) -> str:
            return (
                text
                .replace("<think>", "<unused0>")
                .replace("</think>", "<unused1>")
            )

        if include_multiturn:
            for u, a in zip(user_turns, assistant_turns):
                formatted.append({
                    "instruction": u.strip(),
                    "input": "",
                    "output": replace_think_tokens(a.strip()),
                })
        else:
            formatted.append({
                "instruction": user_turns[0].strip(),
                "input": "",
                "output": replace_think_tokens(assistant_turns[0].strip()),
            })

    if formatted:
        logger.info(f"formatted_ds[0]: {formatted[0]}")

    return formatted


# ---------------------------------------------------------------------
# Unified DataModule
# ---------------------------------------------------------------------

@dataclass
class UnifiedSFTDataModule(DataModule):
    """
    Unified SFT DataModule.

    dataset_type:
      - medical_o1
      - kissanai
    """

    dataset_type: Literal[
        "medical_o1",
        "kissanai",
    ] = "medical_o1"

    mask_prompt: bool = False
    val_split_fraction: float = 0.1
    prompt_style: Union[str, PromptStyle] = "pragna-1b"
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 4
    include_multiturn_conversations: bool = True

    medical_repo_id: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
    kissanai_repo_id: str = "KissanAI/Thinking-climate-100k"

    access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

        if self.dataset_type not in {"medical_o1", "kissanai"}:
            raise ValueError(f"Invalid dataset_type: {self.dataset_type}")

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 2,
        max_seq_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        from datasets import load_dataset

        if self.dataset_type == "medical_o1":
            load_dataset(self.medical_repo_id, "en", token=self.access_token)
        else:
            load_dataset(self.kissanai_repo_id, token=self.access_token)

    def setup(self, stage: str = "") -> None:
        from datasets import load_dataset

        if self.dataset_type == "medical_o1":
            ds = load_dataset(self.medical_repo_id, "en", token=self.access_token)
            data = format_medical_o1(ds["train"])
        else:
            ds = load_dataset(self.kissanai_repo_id, token=self.access_token)
            data = format_kissanai(
                ds["train"],
                self.include_multiturn_conversations,
            )

        train_data, val_data = random_split(
            data,
            [1.0 - self.val_split_fraction, self.val_split_fraction],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.train_dataset = SFTDataset(
            data=list(train_data),
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

        self.test_dataset = SFTDataset(
            data=list(val_data),
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
            ),
        )
