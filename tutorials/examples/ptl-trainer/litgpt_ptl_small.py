# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
import litgpt
from litgpt.lora import GPT
from litgpt.data import Alpaca2k
import lightning as L


class LitLLM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GPT.from_name(
            name="pythia-160m",
            lora_r=8,
            lora_alpha=8,
            lora_dropout=0.05,
            lora_query=True,
            lora_key=False,
            lora_value=True,
        )
        litgpt.lora.mark_only_lora_as_trainable(self.model)

    def setup(self, stage):
        state_dict = torch.load("checkpoints/EleutherAI/pythia-160m/lit_model.pth")
        self.model.load_state_dict(state_dict, strict=False)

    def training_step(self, batch):
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
    tokenizer = litgpt.Tokenizer("checkpoints/EleutherAI/pythia-160m")
    data.connect(tokenizer, batch_size=1, max_seq_length=512)
    trainer = L.Trainer(
        max_epochs=2,
        accumulate_grad_batches=8,
        precision="bf16-true",
    )
    trainer.fit(model, data)
