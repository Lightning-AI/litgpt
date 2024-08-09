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
        self.model, self.tokenizer = self.llm.trainer_setup()

    def training_step(self, batch):
        # TODO: Further abstract the forward pass, maybe llm.trainer_forward(inputs, targets=None)
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]


if __name__ == "__main__":

    model = LitLLM()
    data = Alpaca2k()

    # TODO: think of a better way to provide the tokenizer to the dataset
    data.connect(model.llm.preprocessor.tokenizer, batch_size=1, max_seq_length=512)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",  # TODO: handle device transfer for tokenizer
        max_epochs=2,
        accumulate_grad_batches=8,
        precision="bf16-true",
    )
    trainer.fit(model, data)

    # TODO: Add inference example
