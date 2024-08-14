import numpy as np
import torch
import torch.nn as nn


class DataSelector(nn.Module):
    def __init__(
        self,
        fabric,
        num_datasets,
        embed_dim=64,
        device="cpu",
        initial_weights=None,
        use_bfloat16=True,
    ):
        super().__init__()
        self.device = device
        # TODO: fix the device passing. Fabric seems to be different notation wise?
        if use_bfloat16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.dataset_embeddings = nn.Embedding(
            num_datasets, embed_dim, dtype=self.dtype
        )
        self.dataset_embeddings.to("cuda")
        self.dataset_embeddings = fabric.setup(self.dataset_embeddings)

        self.proj = nn.Linear(embed_dim, 1, dtype=self.dtype)
        self.proj.to("cuda")
        self.proj = fabric.setup(self.proj)

        if initial_weights is not None:
            assert (
                len(initial_weights) == num_datasets
            ), "Initial weights length must equal num datasets"
            with torch.no_grad():
                logits = torch.log(
                    torch.tensor(initial_weights, device="cuda", dtype=self.dtype)
                ) / (1 - torch.tensor(initial_weights, device="cuda", dtype=self.dtype))
                logits = logits.unsqueeze(1)
                self.proj.bias.data = logits
                self.proj.weight.data.fill_(0)

    def forward(self, dataset_ids):
        dataset_embeds = self.dataset_embeddings(dataset_ids)
        return self.proj(dataset_embeds).squeeze(1)

    def update(self, dev_grads, train_grads, dataset_ids):
        similarities = [
            self.compute_gradient_similarity(dev_grads, train_grad)
            for train_grad in train_grads
        ]
        rewards = torch.tensor(similarities, device=dataset_ids.device)

        logits = self(dataset_ids)
        loss = -torch.mean(logits * rewards)  # Policy gradient

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return torch.softmax(logits.detach(), dim=-1)


def basic_linear_scheduler(num_total_iters: int, curr_iter: int, num_other_datasets: int = 1) -> list:
    """
    Linearly decays the sampling rate of pretrain data while increasing other data
    """
    progress = curr_iter / num_total_iters
    return [1 - progress] + [progress / num_other_datasets] 
