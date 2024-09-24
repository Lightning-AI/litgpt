# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from pathlib import Path
import pytest
from tests.conftest import RunIf

import torch
from litgpt.api import LLM
from litgpt.data import Alpaca2k
import lightning as L


REPO_ID = Path("EleutherAI/pythia-14m")


class LitLLM(L.LightningModule):
    def __init__(self, checkpoint_dir, tokenizer_dir=None, trainer_ckpt_path=None):
        super().__init__()

        self.llm = LLM.load(checkpoint_dir, tokenizer_dir=tokenizer_dir, distribute=None)
        self.trainer_ckpt_path = trainer_ckpt_path

    def setup(self, stage):
        self.llm.trainer_setup(trainer_ckpt=self.trainer_ckpt_path)

    def training_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("validation_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]


@pytest.mark.dependency()
def test_download_model():
    LLM.load(
        model="EleutherAI/pythia-14m",
        distribute=None
    )


@pytest.mark.dependency(depends=["test_download_model"])
@RunIf(min_cuda_gpus=1)
def test_usecase1_pretraining_from_random_weights(tmp_path):
    llm = LLM.load("EleutherAI/pythia-14m", tokenizer_dir="EleutherAI/pythia-14m", init="random")
    llm.save("pythia-14m-random-weights")
    del llm

    lit_model = LitLLM(checkpoint_dir="pythia-14m-random-weights", tokenizer_dir="EleutherAI/pythia-14m")
    data = Alpaca2k()

    data.connect(lit_model.llm.tokenizer, batch_size=4, max_seq_length=128)

    trainer = L.Trainer(
        max_epochs=1,
        overfit_batches=2,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    text = lit_model.llm.generate("hello world")
    assert isinstance(text, str)


@pytest.mark.dependency(depends=["test_download_model"])
@RunIf(min_cuda_gpus=1)
def test_usecase2_continued_pretraining_from_checkpoint(tmp_path):
    lit_model = LitLLM(checkpoint_dir="EleutherAI/pythia-14m")
    data = Alpaca2k()

    data.connect(lit_model.llm.tokenizer, batch_size=4, max_seq_length=128)

    trainer = L.Trainer(
        accelerator="cuda",
        max_epochs=1,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    text = lit_model.llm.generate("hello world")
    assert isinstance(text, str)


@pytest.mark.dependency(depends=["test_download_model", "test_usecase2_continued_pretraining_from_checkpoint"])
@RunIf(min_cuda_gpus=1)
def test_usecase3_resume_from_trainer_checkpoint(tmp_path):

    def find_latest_checkpoint(directory):
        latest_checkpoint = None
        latest_time = 0

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.ckpt'):
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_checkpoint = file_path

        return latest_checkpoint

    lit_model = LitLLM(checkpoint_dir="EleutherAI/pythia-14m", trainer_ckpt_path=find_latest_checkpoint("lightning_logs"))

    data = Alpaca2k()
    data.connect(lit_model.llm.tokenizer, batch_size=4, max_seq_length=128)

    trainer = L.Trainer(
        accelerator="cuda",
        max_epochs=1,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    text = lit_model.llm.generate("hello world")
    assert isinstance(text, str)


@pytest.mark.dependency(depends=["test_download_model", "test_usecase2_continued_pretraining_from_checkpoint"])
@RunIf(min_cuda_gpus=1)
def test_usecase4_manually_save_and_resume(tmp_path):

    lit_model = LitLLM(checkpoint_dir="EleutherAI/pythia-14m")
    data = Alpaca2k()

    data.connect(lit_model.llm.tokenizer, batch_size=4, max_seq_length=128)

    trainer = L.Trainer(
        accelerator="cuda",
        max_epochs=1,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    text = lit_model.llm.generate("hello world")
    assert isinstance(text, str)

    lit_model.llm.save("finetuned_checkpoint")

    del lit_model
    lit_model = LitLLM(checkpoint_dir="finetuned_checkpoint")

    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    text = lit_model.llm.generate("hello world")
    assert isinstance(text, str)
