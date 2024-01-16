# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Adapted from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import lightning as L

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config, GPT
from lit_gpt.utils import check_valid_checkpoint_dir, num_parameters, load_checkpoint

# System parameters
devices = 1
gradient_accumulation_iters = 1

# Hyperparameters
learning_rate = 5e-7  # TODO
grad_clip_norm = 10.0  # TODO
epochs = 1
warmup_steps = 150  # TODO
max_length = 512  # TODO
dpo_beta = 0.1  # TODO

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    data_dir: Path = Path("data/alpaca"),  # TODO
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),  # TODO
):
    # TODO configure logger, strategy
    fabric = L.Fabric(devices=devices, strategy="auto", precision="bf16-true")
    fabric.print(hparams)
    fabric.launch(main, checkpoint_dir)


def main(fabric, checkpoint_dir: Path):
    # policy_model = transformers.AutoModelForCausalLM.from_pretrained(
    #     config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True,
    #     torch_dtype=policy_dtype, **model_kwargs)
    # disable_dropout(policy)

    check_valid_checkpoint_dir(checkpoint_dir)
    fabric.seed_everything(0)

    # if fabric.global_rank == 0:
    #     os.makedirs(out_dir, exist_ok=True)
    #
    # train_data = torch.load(data_dir / "train.pt")
    # val_data = torch.load(data_dir / "test.pt")

    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    with fabric.init_module(empty_init=(devices > 1)):
        policy_model = GPT(config)
        reference_model = GPT(config)

    fabric.print(f"Number of trainable parameters: {num_parameters(policy_model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(policy_model, requires_grad=False):,}")

    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=learning_rate)
    policy_model, optimizer = fabric.setup(policy_model, optimizer)
    reference_model = fabric.setup_module(reference_model)
    # TODO: Do we need to disable dropout on the models?

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=(lambda step: min(1.0, (step + 1) / (warmup_steps + 1)))
    )

    load_checkpoint(fabric, policy_model, checkpoint_path)
    fabric.seed_everything(0)  # TODO

    train_time = time.perf_counter()
    train(fabric, policy_model, reference_model, optimizer, scheduler)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f} seconds.")


def train(fabric, policy_model, reference_model, optimizer, scheduler, train_dataloader):
    policy_model.train()
    reference_model.eval()

    iter_count = 0
    step_count = 0

    for batch in train_dataloader:
        # TODO evaluation

        iter_count += 1
        is_accumulating = iter_count % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(policy_model, enabled=is_accumulating):
            policy_chosen_logps, policy_rejected_logps = model_forward(policy_model, batch)
            reference_chosen_logps, reference_rejected_logps = model_forward(reference_model, batch)

            loss = loss_function(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            if step_count > warmup_steps:
                scheduler.step()
            step_count += 1


def model_forward(model: torch.nn.Module, batch):
    # TODO: Concatenate them into one big batch so only one forward/backward is needed
    chosen_logits = model(batch["chosen_input_ids"])  # , attention_mask=concatenated_batch['concatenated_attention_mask'])
    rejected_logits = model(batch["rejected_input_ids"])  # , attention_mask=concatenated_batch['concatenated_attention_mask'])

    # TODO: is .float() needed here?
    chosen_logps = logits_to_logprobs(chosen_logits.float(), batch["labels"])
    rejected_logps = logits_to_logprobs(rejected_logits.float(), batch["labels"])
    return chosen_logps, rejected_logps


def logits_to_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()  # TODO: why clone?
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)  # TODO

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0  # TODO

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)


def loss_function(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
):

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # if reference_free:
    #     ref_logratios = 0

    logits = pi_logratios - ref_logratios

    # Eq. 7 in paper
    loss = -F.logsigmoid(dpo_beta * logits).mean()

    # chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    # rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return loss  # , chosen_rewards, rejected_rewards


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
