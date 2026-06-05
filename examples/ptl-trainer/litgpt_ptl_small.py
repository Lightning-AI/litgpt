# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import lightning as L
import torch

from litgpt import LLM
from litgpt.data import Alpaca2k


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


if __name__ == "__main__":
    batch_size = 8
    accumulate_grad_batches = 1

    #########################################################
    # Use case 1: Pretraining from random weights
    #########################################################

    llm = LLM.load("EleutherAI/pythia-160m", tokenizer_dir="EleutherAI/pythia-160m", init="random")
    llm.save("pythia-160m-random-weights")
    del llm

    lit_model = LitLLM(checkpoint_dir="pythia-160m-random-weights", tokenizer_dir="EleutherAI/pythia-160m")
    data = Alpaca2k()

    data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=1,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.generate("hello world")

    del lit_model

    #############################################################################
    # Use case 2: Continued pretraining / finetuning from downloaded checkpoint
    #############################################################################

    lit_model = LitLLM(checkpoint_dir="EleutherAI/pythia-160m")
    data = Alpaca2k()

    data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=1,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.generate("hello world")

    del lit_model

    #########################################################
    # Use case 3: Resume training from Trainer checkpoint
    #########################################################

    import os

    def find_latest_checkpoint(directory):
        latest_checkpoint = None
        latest_time = 0

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".ckpt"):
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_checkpoint = file_path

        return latest_checkpoint

    lit_model = LitLLM(
        checkpoint_dir="EleutherAI/pythia-160m", trainer_ckpt_path=find_latest_checkpoint("lightning_logs")
    )

    data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=1,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.generate("hello world")

    #################################################################
    # Use case 4: Resume training after saving a checkpoint manually
    #################################################################

    lit_model.llm.save("finetuned_checkpoint")
    del lit_model
    lit_model = LitLLM(checkpoint_dir="finetuned_checkpoint")

    data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=1,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.generate("hello world")
