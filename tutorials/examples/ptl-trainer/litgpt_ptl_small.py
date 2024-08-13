# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
import litgpt
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L


class LitLLM(L.LightningModule):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.llm = LLM.load(checkpoint_dir, distribute=None)

    def setup(self, stage):
        self.llm.trainer_setup()
        # Load an existing checkpoint from a previous Trainer run by providing a `trainer_ckpt`:
        #self.llm.trainer_setup(trainer_ckpt="/teamspace/studios/this_studio/lightning_logs/version_4/checkpoints/epoch=0-step=238.ckpt")
        

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
    
    lit_model = LitLLM(checkpoint_dir="EleutherAI/pythia-160m")
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
    
    #########################################################
    # Save and resume
    lit_model.llm.save("finetuned_checkpoint")
    del lit_model
    lit_model = LitLLM(checkpoint_dir="finetuned_checkpoint")

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
