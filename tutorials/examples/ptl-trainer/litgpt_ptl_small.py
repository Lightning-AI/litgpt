# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
import litgpt
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L


class LitLLM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.llm = LLM.load("EleutherAI/pythia-160m", distribute=None)

    def setup(self, stage):
        self.llm.trainer_setup()

    def training_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    
    lit_model = LitLLM()
    data = Alpaca2k()

    data.connect(lit_model.llm.tokenizer, batch_size=8, max_seq_length=512)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=1,
        #accumulate_grad_batches=8,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    # TODO: Add inference example