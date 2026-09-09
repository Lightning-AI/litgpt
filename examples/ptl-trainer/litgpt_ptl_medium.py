import lightning as L
import torch

import litgpt
from litgpt.data import Alpaca2k
from litgpt.lora import GPT, merge_lora_weights


class LitLLM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GPT.from_name(
            name="Llama-3.1-8B",
            lora_r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_key=False,
            lora_value=True,
        )
        litgpt.lora.mark_only_lora_as_trainable(self.model)

    def on_train_start(self):
        state_dict = torch.load("checkpoints/meta-llama/Meta-Llama-3.1-8B/lit_model.pth", mmap=True)
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
    data = Alpaca2k()
    tokenizer = litgpt.Tokenizer("checkpoints/meta-llama/Meta-Llama-3.1-8B")
    data.connect(tokenizer, batch_size=1, max_seq_length=512)

    trainer = L.Trainer(
        devices=1,
        max_epochs=2,
        accumulate_grad_batches=8,
        precision="bf16-true",
    )
    with trainer.init_module(empty_init=True):
        model = LitLLM()

    trainer.fit(model, data)

    # Save final checkpoint
    merge_lora_weights(model.model)
    trainer.save_checkpoint("checkpoints/finetuned.ckpt", weights_only=True)
