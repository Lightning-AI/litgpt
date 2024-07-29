# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import contextlib
import gc
import math
import pprint
import time
from collections import defaultdict
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import measure_flops, ThroughputMonitor

from litgpt import Tokenizer
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.data import DataModule, MicroLlama, TinyLlama
from litgpt.data.mixed_dataset import CombinedLoader
from litgpt.data_selection.data_selectors import DataSelector
from litgpt.model import Block, CausalSelfAttention, Config, GPT, LLaMAMLP
from litgpt.utils import (
    capture_hparams,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    CycleIterator,
    extend_checkpoint_dir,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
    save_hyperparameters,
)
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from typing_extensions import Literal


@contextlib.contextmanager
def selective_grad_mode(enable_grad):
    if enable_grad:
        with torch.enable_grad():
            yield
    else:
        with torch.no_grad():
            yield


# accumulate grad and return the grad norm
def accumulate_grads(fabric, model, loss):
    fabric.backward(loss)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))


# TODO: backwards update, just a static weight for now
class DataWeights(nn.Module):
    def __init__(self, init_weights: list = [1, 1], normalize: bool = False, **kwargs):
        super(DataWeights, self).__init__()
        self.source_weights = nn.Parameter(
            torch.tensor(init_weights, dtype=torch.float)
        )  # (pretrain_w, sft_w)
        self.normalize = normalize

    def forward(self):
        if self.normalize:
            return torch.softmax(self.source_weights, dim=0)
        else:
            return self.source_weights


def compute_gradient_similarities(gradients, key="_forward_module.lm_head.weight"):
    similarities = {}
    datasets = list(gradients.keys())
    for i, dataset1 in enumerate(datasets):
        similarities[dataset1] = {}
        for dataset2 in datasets[i + 1 :]:
            grad1 = gradients[dataset1][key].view(-1)
            grad2 = gradients[dataset2][key].view(-1)
            sim = torch.nn.functional.cosine_similarity(grad1, grad2, dim=0)
            similarities[dataset1][dataset2] = sim.item()
            if dataset2 not in similarities:
                similarities[dataset2] = {}
            similarities[dataset2][dataset1] = sim.item()
    return similarities


def compute_gradient_similarity(grad1, grad2):
    grad1_flat = grad1.view(-1)
    grad2_flat = grad2.view(-1)
    return F.cosine_similarity(grad1_flat, grad2_flat, dim=0)


def is_distributed_environment():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def round_and_normalize(probs: torch.Tensor, decimals: int = 5) -> torch.Tensor:
    """
    Round the probabilities to a certain number of decimal places and normalize.

    Args:
    probs (torch.Tensor): Input probability tensor
    decimals (int): Number of decimal places to round to

    Returns:
    torch.Tensor: Rounded and normalized probabilities
    """
    factor = 10**decimals
    probs = torch.round(probs * factor) / factor
    return probs / probs.sum()


def compute_lm_loss(model, input_ids, targets):
    logits = model(input_ids)
    lm_loss = chunked_cross_entropy(logits, targets)
    return lm_loss


def compute_sft_loss(model, input_ids, targets):
    logits = model(input_ids)
    sft_loss = chunked_cross_entropy(
        logits[..., :-1, :], targets[..., 1:], chunk_size=0
    )
    return sft_loss


def setup(
    model_name: str,
    model_config: Optional[Config] = None,
    out_dir: Path = Path("out/pretrain"),
    logs_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Path] = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        max_norm=1.0,
        min_lr=4e-5,
        tie_embeddings=False,
        max_steps=7500,
        lr_warmup_fraction=0.01,
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    optimizer: Union[str, Dict] = "AdamW",
    devices: Union[int, str] = "auto",
    tokenizer_dir: Optional[Path] = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "tensorboard",
    seed: int = 42,
):
    """Pretrain a model.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Use "list" to list the supported models.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``. Overrides the `model_name` if specified.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Determines a compatible precision setting by default.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
            Useful for continued pretraining. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.

        devices: How many devices/GPUs to use. Uses all GPUs by default.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
    """
    if model_name == "list":
        available_models = "\n".join(sorted(name_to_config))
        print(f"Available values:\n{available_models}")
        quit()

    if initial_checkpoint_dir is not None:
        initial_checkpoint_dir = extend_checkpoint_dir(initial_checkpoint_dir)

    if tokenizer_dir is not None:
        tokenizer_dir = extend_checkpoint_dir(tokenizer_dir)

    if model_config is None:
        # Support both model_name options: meta-llama/Meta-Llama-3-8B & Meta-Llama-3-8B
        try:
            model_config = Config.from_name(model_name)
        except ValueError:
            print(f"Model name {model_name} is not supported.\n")
            available_models = "\n".join(sorted(name_to_config))
            print(f"Available values:\n{available_models}")
            quit()

    hparams = capture_hparams()
    data = TinyLlama() if data is None else data

    config = Config.from_name(model_name) if model_config is None else model_config
    precision = precision or get_default_supported_precision(training=True)
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)
    logs_dir = init_out_dir(logs_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    if train.episode_length is None:
        train.episode_length = float("inf")

    assert train.episode_length > 0, "Episode length must be positive"

    logger = choose_logger(
        logger_name,
        logs_dir,
        name=f"pretrain-{config.name}",
        resume=resume,
        log_interval=train.log_interval,
    )

    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            state_dict_type="full",
            sharding_strategy="HYBRID_SHARD",
        )
    else:
        strategy = "auto"
    fabric = L.Fabric(
        devices=devices, strategy=strategy, precision=precision, loggers=[logger]
    )
    fabric.launch()

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(hparams)

    main(
        fabric,
        devices,
        seed,
        initial_checkpoint_dir,
        resume,
        config,
        data,
        out_dir,
        tokenizer_dir,
        tokenizer,
        train,
        eval,
        optimizer,
        train.max_steps,
        train.episode_length,
        precision,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Optional[Path],
    resume: Union[bool, Path],
    config: Config,
    data: DataModule,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    tokenizer: Optional[Tokenizer],
    train: TrainArgs,
    eval: EvalArgs,
    optimizer_config: Union[str, Dict],
    max_steps: int,
    episode_length: int,
    precision: str,
) -> None:
    validate_args(train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
    if train.max_seq_length:
        model.max_seq_length = train.max_seq_length

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    # model = torch.compile(model)
    model = fabric.setup(model)

    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    optimizer = instantiate_torch_optimizer(
        optimizer_config, model.parameters(), **extra_kwargs
    )
    optimizer = fabric.setup_optimizers(optimizer)

    # need max iters up here to evenly space
    if max_steps is None:
        max_tokens_per_device = train.max_tokens // fabric.world_size
        tokens_per_iter = train.micro_batch_size * model.max_seq_length
        max_iters = max_tokens_per_device // tokens_per_iter
    else:
        max_iters = max_steps

    train_dataloader, val_dataloader = get_dataloaders(
        fabric, data, tokenizer, train, model.max_seq_length, max_iters=max_iters
    )

    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    state = {
        "model": model,
        "optimizer": optimizer,
        # "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }

    # need dummy optimizers to get the gradients from each dataset separately.
    if data.use_adaptive_sampling:
        lm_dummy_optimizer = instantiate_torch_optimizer(
            optimizer_config, model.parameters(), **extra_kwargs
        )
        lm_dummy_optimizer = fabric.setup_optimizers(lm_dummy_optimizer)
        sft_dummy_optimizers = {}
        for sft_dataset in train_dataloader.loaders:
            sft_dummy_optimizers[sft_dataset] = instantiate_torch_optimizer(
                optimizer_config, model.parameters(), **extra_kwargs
            )
            sft_dummy_optimizers[sft_dataset] = fabric.setup_optimizers(
                sft_dummy_optimizers[sft_dataset]
            )

        state["lm_optimizer_dummy"] = lm_dummy_optimizer
        state["sft_optimizers_dummy"] = sft_dummy_optimizers

    if resume is True:
        resume = max(
            out_dir.rglob("step-*/*.pth"),
            key=(lambda p: int(p.parent.name.split("-")[1])),
        )
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    fit(
        fabric,
        devices,
        state,
        train_dataloader,
        val_dataloader,
        out_dir,
        tokenizer_dir,
        train,
        eval,
        max_iters,
        episode_length,
        precision,
        data.use_adaptive_sampling,
    )

    # Save final checkpoint
    save_checkpoint(fabric, state, tokenizer_dir, out_dir / "final" / "lit_model.pth")

    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def collect_gradients(fabric, model, loss):
    gradients = {}

    def make_hook(name):
        def hook(grad):
            gradients[name] = grad.clone().detach()
            return grad

        return hook

    handles = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            handle = param.register_hook(make_hook(name))
            handles.append(handle)

    fabric.backward(loss)

    for handle in handles:
        handle.remove()

    return gradients


def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    train: TrainArgs,
    eval: EvalArgs,
    max_iters: int,
    episode_length: int,
    precision: str,
    use_adaptive_sampling: bool,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    if use_adaptive_sampling:
        lm_dummy_optimizer = state["lm_optimizer_dummy"]
        sft_dummy_optimizers = state["sft_optimizers_dummy"]

    # store the data selector in the checkpoint for resuming
    if "data_selector" not in state:
        data_selector = DataSelector(
            fabric,
            len(train_dataloader.loaders),
            device=devices,
            initial_weights=[0.85, 0.05, 0.05, 0.05],
            use_bfloat16=precision == "bf16-true",
        )
        data_optimizer = torch.optim.AdamW(data_selector.parameters(), lr=1e-3)
        state["data_selector"] = data_selector
        state["data_optimizer"] = data_optimizer
    else:
        data_selector = state["data_selector"]
        data_optimizer = state["data_optimizer"]

    if eval.initial_validation:
        val_loss, val_loss_lm, val_loss_sft, _ = validate(
            fabric, model, val_dataloader, max_iters=eval.max_iters
        )
        val_loss = f"{val_loss:.3f}"
        val_loss_lm = f"{val_loss_lm:.3f}"
        val_loss_sft = f"{val_loss_sft:.3f}"
    else:
        validate(
            fabric, model, val_dataloader, max_iters=2, do_collect_gradients=True
        )  # sanity check
        val_loss = "n/a"

    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))
        model_fwd = lambda: meta_model(x)
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(
            f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}"
        )
        del meta_model, x

    if max_iters is None:
        max_tokens_per_device = train.max_tokens // fabric.world_size
        tokens_per_iter = train.micro_batch_size * model.max_seq_length
        max_iters = max_tokens_per_device // tokens_per_iter

    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices)
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss_total = RunningMean(
        window=train.gradient_accumulation_iters(devices), sync_on_compute=False
    ).to(fabric.device)
    lm_loss_total = RunningMean(
        window=train.gradient_accumulation_iters(devices), sync_on_compute=False
    ).to(fabric.device)
    sft_loss_total = RunningMean(
        window=train.gradient_accumulation_iters(devices), sync_on_compute=False
    ).to(fabric.device)

    sft_loss_per_dataset = {}

    fabric.barrier()
    total_t0 = time.perf_counter()
    warmup_iters = train.warmup_iters(devices, max_iters, train_dataloader)

    # train grads for each dataset, for adjusting sampling
    data_weights = None
    train_gradients = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    train_gradient_counts = defaultdict(int)
    first_iter = True

    for train_data in train_iterator:
        if first_iter:
            optimizer.zero_grad()
            # Reset previous_grads and train_gradients at the start of each accumulation cycle
            previous_grads = {
                name: torch.zeros_like(param)
                for name, param in model.named_parameters()
                if name == "_forward_module.lm_head.weight"
                or name == "_forward_module.\_fsdp_wrapped_module.lm_head.weight"
            }
            train_gradients = defaultdict(lambda: defaultdict(torch.zeros_like))
            first_iter = False

        if state["iter_num"] >= max_iters:
            break

        if state["iter_num"] % episode_length == 0 and state["iter_num"] != 0:
            # Update the sampling ratios

            dev_loss, dev_loss_lm, dev_loss_sft, dev_grads = validate(
                fabric,
                model,
                val_dataloader,
                max_iters=eval.max_iters,
                do_collect_gradients=True,
            )

            metrics = {
                "val_loss": dev_loss,
                "val_loss_lm": dev_loss_lm,
                "val_loss_sft": dev_loss_sft,
                "val_ppl": math.exp(dev_loss),
            }
            fabric.log_dict(metrics, step=state["iter_num"] - 1)
            fabric.barrier()
            layer_key = (
                "_forward_module._fsdp_wrapped_module.lm_head.weight"
                if is_distributed_environment()
                else "_forward_module.lm_head.weight"
            )

            if state["iter_num"] == episode_length:
                # first iter, print the similarities of dev grads to each other
                fabric.print("Dev grad similarities:")
                fabric.print(compute_gradient_similarities(dev_grads, key=layer_key))

            if not train.freeze_sampling_rate:
                similarities = {}

                # if we have not accumulated the train gradients yet, skip updating the sampling rate
                if len(train_gradients) == 0:
                    fabric.print(
                        "Have not hit first gradient accumulation yet, waiting to update the sampling rate...."
                    )
                else:
                    # TODO: wtf why is layer_key undefined here, investigate later
                    if is_distributed_environment():
                        try:
                            mean_dev_gradient = torch.mean(
                                torch.stack(
                                    [
                                        g[
                                            "_forward_module._fsdp_wrapped_module.lm_head.weight"
                                        ]
                                        for g in dev_grads.values()
                                        if layer_key in g
                                    ]
                                ),
                                dim=0,
                            )
                        except:
                            torch.distributed.barrier()
                            if torch.distributed.get_rank() == 0:
                                breakpoint()
                            else:
                                time.sleep(10000)
                    else:
                        mean_dev_gradient = torch.mean(
                            torch.stack(
                                [
                                    g["_forward_module.lm_head.weight"]
                                    for g in dev_grads.values()
                                    if layer_key in g
                                ]
                            ),
                            dim=0,
                        )

                    for dataset_name, grad in train_gradients.items():
                        if grad is not None:
                            sim = compute_gradient_similarity(
                                mean_dev_gradient,
                                grad[layer_key],
                            )
                            similarities[dataset_name] = sim.item()
                        else:
                            similarities[dataset_name] = 0

                    dataset_ids = torch.arange(
                        len(train_dataloader.loaders), device=fabric.device
                    )

                    try:
                        rewards = torch.tensor(
                            [
                                similarities[name]
                                for name in train_dataloader.loaders.keys()
                            ],
                            device=fabric.device,
                        )
                    except:
                        torch.distributed.barrier()
                        if torch.distributed.get_rank() == 0:
                            breakpoint()
                        else:
                            time.sleep(10000)

                    logits = data_selector(dataset_ids)
                    data_loss = -torch.mean(logits * rewards)  # Policy gradient loss

                    data_optimizer.zero_grad()
                    fabric.backward(data_loss)
                    data_optimizer.step()

                    del similarities
                    del mean_dev_gradient
                    del data_loss

                    # Get new sampling rates from updated LanguageActor
                    with torch.no_grad():
                        logits = data_selector(dataset_ids)
                        new_sampling_rate = torch.softmax(logits, dim=-1)
                        new_sampling_rate = round_and_normalize(
                            new_sampling_rate
                        ).tolist()
            else:
                new_sampling_rate = train_dataloader.sampling_rates
                rewards = torch.zeros(len(train_dataloader.loaders))

            # Update sampling rates in the dataloader
            fabric.print(f"Old sampling rates: {train_dataloader.sampling_rates}")
            train_dataloader.sampling_rates = new_sampling_rate
            train_dataloader._dataloader.sampling_rates = new_sampling_rate

            fabric.print(f"Updated sampling rates: {new_sampling_rate}")

            sampling_rates_dict = {
                f"sampling_rate_{name}": round(sampling_rate, 3)
                for name, sampling_rate in zip(
                    train_dataloader.loaders.keys(), new_sampling_rate
                )
            }
            metrics = {
                **sampling_rates_dict,
                "reward": rewards.sum().item(),
            }
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

            # Reset train gradients for the next episode
            # del train_gradients
            # del train_gradient_counts
            # del logits
            # del new_sampling_rate
            # train_gradients = defaultdict(
            # lambda: defaultdict(lambda: defaultdict(float))
            # )
            # train_gradient_counts = defaultdict(int)

            gc.collect()
            torch.cuda.empty_cache()

        if train.decay_lr:
            lr = get_lr_decay_stage(
                optimizer["lr"], state["iter_num"], initial_iter, train.min_lr
            )
        else:
            # determine and set the learning rate for this iteration
            lr = get_lr(
                2e-5,  # the default LR is too high. Using this one
                state["iter_num"],
                warmup_iters,
                max_iters,
                train.min_lr,
            )

        assert lr >= 0, "Learning rate must be positive"

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        if isinstance(train_data, tuple):
            # structure: ({"sft": sft_data, "lm": lm_data}, batch_idx, dataloader_idx)
            paired_data = train_data[0]
            sft_datasets = {
                key: paired_data[key] for key in paired_data.keys() if key != "lm"
            }
            lm_data = paired_data.get("lm", None)
            batch_idx = train_data[1]
            dataloader_idx = train_data[2]

        elif isinstance(train_data, dict):
            lm_data = train_data.get("lm", None)
            sft_datasets = {
                key: train_data[key] for key in train_data.keys() if key != "lm"
            }

        if data_weights is None:
            data_weights = DataWeights([0.85, 0.05, 0.05, 0.05])

        # Process LM data
        is_accumulating = (
            state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        )

        all_losses = []

        lm_samples = 0

        if lm_data is not None:
            input_ids = lm_data[:, 0 : model.max_seq_length].contiguous().long()
            targets = lm_data[:, 1 : (model.max_seq_length + 1)].contiguous().long()
            lm_loss = compute_lm_loss(model, input_ids, targets)

            all_losses.append(lm_loss)
            lm_loss_total.update(lm_loss.detach())
            lm_samples += input_ids.size(0)

            if use_adaptive_sampling:
                model.zero_grad()
                lm_loss.backward(retain_graph=True)
                for name, param in model.named_parameters():
                    if param.grad is not None and (
                        name == "_forward_module.lm_head.weight"
                        or name == "_forward_module._fsdp_wrapped_module.lm_head.weight"
                    ):
                        if name not in train_gradients["lm"]:
                            # very important: we can't subtract the previous grads the FIRST time since it will be equal to train_grads, leading to all 0s
                            train_gradients["lm"][name] = param.grad
                        else:
                            train_gradients["lm"][name] += (
                                param.grad - previous_grads[name].detach()
                            )
                        previous_grads[name] = param.grad.clone()

        # process SFT data
        sft_samples_per_dataset = {}

        for i, key in enumerate(sft_datasets):
            sft_data = sft_datasets[key]
            if sft_data is not None:
                input_ids, targets = sft_data["input_ids"], sft_data["labels"]
                sft_loss = compute_sft_loss(model, input_ids, targets)

                if key not in sft_loss_per_dataset:
                    sft_loss_per_dataset[key] = RunningMean(
                        window=train.gradient_accumulation_iters(devices),
                        sync_on_compute=False,
                    ).to(fabric.device)

                sft_loss_per_dataset[key].update(sft_loss.detach())
                running_loss_total.update(sft_loss.detach())
                sft_loss_total.update(sft_loss.detach())
                all_losses.append(sft_loss)
                sft_samples_per_dataset[key] = sft_samples_per_dataset.get(
                    key, 0
                ) + input_ids.size(0)

                if use_adaptive_sampling:
                    model.zero_grad()
                    sft_loss.backward(retain_graph=True)
                    for name, param in model.named_parameters():
                        if param.grad is not None and (
                            name == "_forward_module.lm_head.weight"
                            or name
                            == "_forward_module._fsdp_wrapped_module.lm_head.weight"
                        ):
                            if name not in train_gradients[key]:
                                train_gradients[key][name] = param.grad
                            else:
                                train_gradients[key][name] += (
                                    param.grad - previous_grads[name].detach()
                                )
                            previous_grads[name] = param.grad.clone()

        loss = sum(all_losses) / len(
            all_losses
        )  # TODO: losses can be combined in different ways.

        # backprop
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        if not is_accumulating:
            previous_grads = {
                name: torch.zeros_like(param)
                for name, param in model.named_parameters()
                if name == "_forward_module.lm_head.weight"
                or name == "_forward_module._fsdp_wrapped_module.lm_head.weight"
            }
            train_gradients = defaultdict(
                lambda: defaultdict(lambda: defaultdict(float))
            )
            train_gradient_counts = defaultdict(int)

            fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        running_loss_total.update(loss.detach())

        if state["iter_num"] % log_iter_interval == 0:
            loss = (
                running_loss_total.compute().item()
            )  # expensive device-to-host synchronization
            loss_lm = lm_loss_total.compute().item()
            avg_loss_lm = loss_lm / lm_samples if lm_samples > 0 else None
            loss_sft = sft_loss_total.compute().item()

            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(
                    state["iter_num"] * train.micro_batch_size * model.max_seq_length
                ),
            )
            sft_loss_breakdown = {
                key: loss_per_dataset.compute().item()
                for key, loss_per_dataset in sft_loss_per_dataset.items()
            }
            avg_loss_sft = {
                key: (
                    loss_per_dataset.compute().item() / sft_samples_per_dataset[key]
                    if sft_samples_per_dataset.get(key, 0) > 0
                    else None
                )
                for key, loss_per_dataset in sft_loss_per_dataset.items()
            }

            sft_num_samples = {
                key: sft_samples_per_dataset.get(key, 0)
                for key in sft_loss_per_dataset.keys()
            }
            metrics = {
                "loss": loss,
                "loss_lm": loss_lm,
                "loss_sft": loss_sft,
                **sft_loss_breakdown,
                "num_samples_lm": lm_samples,
                **sft_num_samples,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0)
                    / (state["iter_num"] - initial_iter)
                    * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"]
                * train.micro_batch_size
                * model.max_seq_length,
                "total_tokens": (
                    state["iter_num"]
                    * train.micro_batch_size
                    * model.max_seq_length
                    * fabric.world_size
                ),
                "learning_rate": lr,
            }
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.3f}"

            sft_sets_str = " | ".join(
                [f"loss {name}: {loss}" for name, loss in sft_loss_breakdown.items()]
            )
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" loss train (lm): {metrics['loss_lm']}",
                f" loss train (sft): {metrics['loss_sft']}",
                f" ~~ breakdown: {sft_sets_str} ~~"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}",
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

        if (
            val_dataloader is not None
            and not is_accumulating
            and state["step_count"] % eval.interval == 0
        ):
            t0 = time.perf_counter()
            val_loss, val_loss_lm, val_loss_sft, mean_grads = validate(
                fabric, model, val_dataloader, max_iters=eval.max_iters
            )

            val_loss = val_loss.item()
            val_loss_lm = val_loss_lm.item()
            val_loss_sft = val_loss_sft.item()
            td = time.perf_counter() - t0

            fabric.print(
                f"iter {state['iter_num']}: val loss {val_loss:.4f}, (lm subset): {val_loss_lm:.4f}, (sft subset): {val_loss_sft:.4f}, val time: {td * 1000:.2f} ms"
            )
            metrics = {
                "val_loss": val_loss,
                "val_loss_lm": val_loss_lm,
                "val_loss_sft": val_loss_sft,
                "val_ppl": math.exp(val_loss),
            }
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

            del val_loss_lm
            del val_loss_sft
            gc.collect()
            torch.cuda.empty_cache()
            fabric.barrier()

        if (
            train.save_interval is not None
            and not is_accumulating
            and state["step_count"] % train.save_interval == 0
        ):
            save_checkpoint(
                fabric,
                state,
                tokenizer_dir,
                out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth",
            )

    # Final validation
    if eval.final_validation:
        val_loss, val_loss_lm, val_loss_sft, _ = validate(
            fabric, model, val_dataloader, max_iters=eval.max_iters
        )
        metrics = {
            "val_loss": val_loss,
            "val_loss_lm": val_loss_lm,
            "val_loss_sft": val_loss_sft,
            "val_ppl": math.exp(val_loss),
        }
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(
            f"Final evaluation | val loss: {val_loss.item():.3f} (lm): {val_loss_lm.item():.3f} (sft): {val_loss_sft.item():.3f} | val ppl: {math.exp(val_loss):.3f}"
        )


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: nn.Module,
    val_dataloader: DataLoader,
    max_iters: int,
    do_collect_gradients: bool = False,
) -> torch.Tensor:
    fabric.barrier()
    fabric.print("Validating ...")
    model.eval()

    losses = []
    losses_lm = []
    losses_sft = []
    gradient_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    gradient_counts = defaultdict(int)

    # print(torch.cuda.memory_summary())

    with selective_grad_mode(do_collect_gradients):
        for k, batch in enumerate(val_dataloader):
            if k >= max_iters:
                break

            if isinstance(
                batch, tuple
            ):  # TODO: this iterator is structured weirdly, maybe reconsider.
                # structure: ({"sft": sft_data, "lm": lm_data}, batch_idx, dataloader_idx)
                paired_data = batch[0]
                lm_data = paired_data.get("lm", None)
                sft_datasets = {
                    key: paired_data[key] for key in paired_data.keys() if key != "lm"
                }
                batch_idx = batch[1]
                dataloader_idx = batch[2]

            model.zero_grad()

            # Process LM data
            if lm_data is not None:
                input_ids = lm_data[:, 0 : model.max_seq_length].contiguous().long()
                targets = lm_data[:, 1 : (model.max_seq_length + 1)].contiguous().long()
                logits = model(input_ids)
                loss = chunked_cross_entropy(logits, targets)
                losses.append(loss.detach())
                losses_lm.append(loss.detach())

                if do_collect_gradients:
                    accumulate_grads(fabric, model, loss)
                    gradient_counts["lm"] += 1

            for name, param in model.named_parameters():
                if param.grad is not None:
                    # TODO: oom issues, only store this for now
                    if (
                        name != "_forward_module.lm_head.weight"
                        and name
                        != "_forward_module._fsdp_wrapped_module.lm_head.weight"
                    ):
                        continue

                    if name not in gradient_sums["lm"]:
                        gradient_sums["lm"][name] = param.grad.clone().detach()
                    else:
                        gradient_sums["lm"][name] += param.grad.clone().detach()

            model.zero_grad()

            # Process SFT data
            for key in sft_datasets:
                sft_data = sft_datasets[key]
                if sft_data is not None:
                    input_ids, targets = sft_data["input_ids"], sft_data["labels"]
                    logits = model(input_ids)
                    sft_loss = chunked_cross_entropy(
                        logits[..., :-1, :], targets[..., 1:], chunk_size=0
                    )

                    losses.append(sft_loss.detach())
                    losses_sft.append(sft_loss.detach())

                    if do_collect_gradients:
                        accumulate_grads(fabric, model, sft_loss)
                        gradient_counts[key] += 1

                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if (
                                name != "_forward_module.lm_head.weight"
                                and name
                                != "_forward_module._fsdp_wrapped_module.lm_head.weight"
                            ):
                                continue
                            if name not in gradient_sums[key]:
                                gradient_sums[key][name] = param.grad.detach().clone()
                            else:
                                gradient_sums[key][name] += param.grad.detach().clone()

                    model.zero_grad()

    val_loss = torch.stack(losses).mean()
    val_loss_lm = torch.stack(losses_lm).mean()
    val_loss_sft = torch.stack(losses_sft).mean()

    mean_gradients = {}
    if do_collect_gradients:
        for task in gradient_sums:
            mean_gradients[task] = {
                name: grad_sum / gradient_counts[task]
                for name, grad_sum in gradient_sums[task].items()
            }

    model.train()
    fabric.barrier()

    del gradient_sums
    del gradient_counts
    del losses
    del losses_lm
    del losses_sft

    gc.collect()
    torch.cuda.empty_cache()

    return val_loss, val_loss_lm, val_loss_sft, mean_gradients


def get_dataloaders(
    fabric: L.Fabric,
    data: DataModule,
    tokenizer: Tokenizer,
    train: TrainArgs,
    block_size: int,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    data.connect(
        tokenizer=tokenizer,
        batch_size=train.micro_batch_size,
        max_seq_length=block_size,
        **kwargs,
    )
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    # this is a quirk of the combined dataloader implementation - the dataloader has no len defined until the first iter()
    if isinstance(train_dataloader, CombinedLoader):
        iter(train_dataloader)

    if isinstance(val_dataloader, CombinedLoader):
        iter(val_dataloader)

    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(
    learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float
) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_lr_decay_stage(
    learning_rate: float, it: int, decay_start: int, min_lr: float
) -> float:
    # learning_rate: the max lr to start from.
    # it: current iter
    # decay_start: the iter at which decay begins
    # min_lr: the minimum LR
    if it < decay_start:
        return learning_rate
    decay_ratio = 0.5 ** ((it - decay_start) / decay_start)
    decayed_lr = learning_rate * decay_ratio
    return max(min_lr, decayed_lr)


def get_lr_linear_decay(
    learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float
) -> float:
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    return learning_rate * (1 - it / max_iters) + min_lr


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(
                init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd)
            )

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(
                init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer)
            )

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)


def save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file):
    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    fabric.save(checkpoint_file, state)
    if fabric.global_rank == 0:
        save_hyperparameters(setup, checkpoint_file.parent)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


def validate_args(
    train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume
) -> None:
    issues = []
    unsupported = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(
                    f"{__file__} doesn't support the {name!r} argument. This is set in {args}"
                )
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(
                    f"{__file__} requires the {name!r} argument. This is set in {args}"
                )
    if initial_checkpoint_dir and resume:
        issues.append(
            "Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one."
        )
    if issues:
        raise ValueError("\n".join(issues))
