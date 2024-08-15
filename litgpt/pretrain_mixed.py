# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import ast
import contextlib
import gc
import math
import pprint
import time
from collections import defaultdict
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from litgpt.data_selection.data_selectors import DataSelector, basic_linear_scheduler
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
    round_and_normalize
)
from litgpt.schedulers import get_lr, get_lr_decay_stage, get_lr_linear_decay
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from torch.quasirandom import SobolEngine

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

class DistributedMovingWindowGP:
    def __init__(self, fabric, n_datasets=2, window_size=100):
        self.fabric = fabric
        self.window_size = window_size
        self.X = torch.zeros((0, n_datasets), dtype=torch.float64, device=fabric.device)
        self.Y = torch.zeros((0, 1), dtype=torch.float64, device=fabric.device)

    def add_observation(self, x, y):
        # Gather observations from all processes
        x_list = [torch.zeros_like(x) for _ in range(self.fabric.world_size)]
        y_list = [torch.zeros_like(y) for _ in range(self.fabric.world_size)]
        
        dist.all_gather(x_list, x)
        dist.all_gather(y_list, y)
        
        x_gathered = torch.stack(x_list)
        y_gathered = torch.stack(y_list)
        
        self.X = torch.cat([self.X, x_gathered], dim=0)
        self.Y = torch.cat([self.Y, y_gathered], dim=0)
        
        if len(self.X) > self.window_size:
            self.X = self.X[-self.window_size:]
            self.Y = self.Y[-self.window_size:]

    def get_data(self):
        return self.X, self.Y

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


def decode_batch(batch, tokenizer):
    # Assuming 'input_ids' is the key for encoded sequences in your batch
    input_ids = batch['input_ids']
    
    # Convert to list of lists
    input_ids_list = input_ids.tolist()
    
    # Decode each sequence
    decoded_texts = []
    for seq in input_ids_list:
        # Remove any padding tokens (usually represented by 0)
        seq = [token for token in seq if token != 0]
        decoded_text = tokenizer.decode(seq)
        decoded_texts.append(decoded_text)
    
    return decoded_texts

def generate_ts_candidates(X, Y, batch_size, n_candidates):
    likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[-1]))

    model = SingleTaskGP(X, Y, likelihood=likelihood, covar_module=covar_module)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    #sobol = SobolEngine(X.shape[-1], scramble=True)
    #X_cand = sobol.draw(n_candidates).to(dtype=X.dtype, device=X.device)
    X_cand = sample_from_simplex(n_candidates, X.shape[-1]).to(dtype=X.dtype, device=X.device)

    #X_cand = project_to_simplex(X_cand)

    with torch.no_grad():
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

    return X_next

def sample_from_simplex(n_samples: int, n_dimensions: int) -> torch.Tensor:
    """Sample from the unit simplex in dim dimensions.

    This is a uniform sample from the unit simplex in dim dimensions.
    """
    samples = torch.distributions.Exponential(rate=torch.ones(n_dimensions)).sample((n_samples,))
    
    # Normalize to make sure each sample sums to 1
    return samples / samples.sum(dim=1, keepdim=True)

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
        max_iters=100000,
        max_additional_steps=None,
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

    if data.initial_sampling_rates:
        initial_sampling_rates = data.initial_sampling_rates
    else:
        initial_sampling_rates = None

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
        train.max_iters,
        train.max_additional_steps,
        train.episode_length,
        precision,
        initial_sampling_rates,
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
    max_iters: int,
    max_additional_steps: Optional[int],
    episode_length: int,
    precision: str,
    initial_sampling_rates: Optional[List[float]],
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
    if max_iters is None:
        max_tokens_per_device = train.max_tokens // fabric.world_size
        tokens_per_iter = train.micro_batch_size * model.max_seq_length
        max_iters = max_tokens_per_device // tokens_per_iter


    train_dataloader, val_dataloader = get_dataloaders(
        fabric,
        data,
        tokenizer,
        train,
        model.max_seq_length,
        max_iters=max_iters,
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

    if resume is True:
        resume = max(
            out_dir.rglob("step-*/*.pth"),
            key=(lambda p: int(p.parent.name.split("-")[1])),
        )
    if resume:
        fabric.print(f"Resuming training from {resume}")
        start_time = time.time()
        fabric.load(resume, state, strict=False)
        end_time = time.time()
        fabric.print("Time to load model: ", end_time - start_time)

    train_time = time.perf_counter()
    fit(
        tokenizer,
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
        max_additional_steps,
        episode_length,
        precision,
        data.use_adaptive_sampling,
        initial_sampling_rates
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
    tokenizer: None, #TODO: remove
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
    max_additional_steps: int,
    episode_length: int,
    precision: str,
    use_adaptive_sampling: bool,
    initial_sampling_rates: Optional[List[float]] = None,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    # store the data selector in the checkpoint for resuming

    if not initial_sampling_rates:
        initial_sampling_rates = [
            1 / len(train_dataloader.loaders) for _ in train_dataloader.loaders
        ]

    # note: we cannot save the data optimizer in the same state due to this line: https://github.com/Lightning-AI/pytorch-lightning/blob/f5d82504da252a255df268bd2bb99bf3f2886d27/src/lightning/fabric/strategies/fsdp.py#L460
    data_selector = DataSelector(
        fabric,
        len(train_dataloader.loaders),
        device=devices,
        initial_weights=initial_sampling_rates,
        use_bfloat16=precision == "bf16-true",
    )
    data_optimizer = torch.optim.AdamW(data_selector.parameters(), lr=1e-3)
    # state["data_selector"] = data_selector
    # state["data_optimizer"] = data_optimizer

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


    if not max_iters and not max_additional_steps:
        max_tokens_per_device = train.max_tokens // fabric.world_size
        tokens_per_iter = train.micro_batch_size * model.max_seq_length
        max_iters = max_tokens_per_device // tokens_per_iter


    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices)
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    if max_iters and max_additional_steps:
        fabric.print("Specified both max iters and max additional steps, overriding max_iters with max_additional_steps")
        max_iters = initial_iter + (max_additional_steps * train.gradient_accumulation_iters(devices))
        fabric.print(f"Setting max_iters to {max_iters}")


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
    initial_lr = optimizer.param_groups[0]["lr"]

    # train grads for each dataset, for adjusting sampling
    train_gradients = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    train_gradient_counts = defaultdict(int)
    first_iter = True
    initial_steps = state["step_count"]


    # keep track of observations for thompson sampling
    if train.scheduler == "ts_gp":
        n_datasets = len(train_dataloader.loaders)
        X = torch.zeros((0, n_datasets), dtype=torch.float64, device=fabric.device)
        Y = torch.zeros((0, 1), dtype=torch.float64, device=fabric.device)  # Change to (0, 1)
        running_avg_performance = None
        alpha = 0.1  # Exponential moving average factor
        INITIAL_EXPLORE_STEPS = 100 # TODO: add in initial exploration phase

        moving_window_gp = DistributedMovingWindowGP(fabric, n_datasets)


    for train_data in train_iterator:
        if first_iter:
            optimizer.zero_grad()
            # Reset previous_grads and train_gradients at the start of each accumulation cycle
            previous_grads = {
                name: torch.zeros_like(param)
                for name, param in model.named_parameters()
                if name == "_forward_module.lm_head.weight"
                or name == "_forward_module._fsdp_wrapped_module.lm_head.weight"
            }
            train_gradients = defaultdict(lambda: defaultdict(torch.zeros_like))
            first_iter = False

        if state["iter_num"] >= max_iters:
            fabric.print("Reached max iters, exiting")
            break
        if state["step_count"] - initial_steps >= max_additional_steps:
            fabric.print("Reached max additional steps, exiting")
            break

        if (
            state["iter_num"] % episode_length == 0 and state["iter_num"] != 0
        ) and not train.freeze_sampling_rate:
            # Update the sampling ratios

            if train.data_scheduler == "grad":
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

                new_sampling_rate = train_dataloader.sampling_rates
                rewards = torch.zeros(len(train_dataloader.loaders))

                if not train.freeze_sampling_rate:
                    similarities = {}
                    #TODO: make this a switch based on train.scheduler

                    # if we have not accumulated the train gradients yet, skip updating the sampling rate
                    if len(train_gradients) == 0:
                        fabric.print(
                            "Have not hit first gradient accumulation yet, waiting to update the sampling rate...."
                        )
                    else:
                        # TODO: wtf why is layer_key undefined here, investigate later
                        if is_distributed_environment():
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
                                    (
                                        similarities[name]
                                        if name in train_dataloader.loaders.keys()
                                        else 0
                                    )
                                    for name in train_dataloader.loaders.keys()
                                ],
                                device=fabric.device,
                            )
                        except:
                            breakpoint()

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
            elif train.data_scheduler == "linear":
                new_sampling_rate = basic_linear_scheduler(max_iters - initial_iter, state["iter_num"] - initial_iter, num_other_datasets=len(train_dataloader.loaders) - 1)
            elif train.data_scheduler == "ts_gp":
                dev_loss, dev_loss_lm, dev_loss_sft, dev_grads = validate(
                    fabric,
                    model,
                    val_dataloader,
                    max_iters=eval.max_iters,
                    do_collect_gradients=True,
                )

                if running_avg_performance is None:
                    running_avg_performance = 0.9 * dev_loss_lm.item() + 0.1 * dev_loss_sft.item()
                else:
                    running_avg_performance = (1 - alpha) * running_avg_performance + alpha * ( 0.9 * dev_loss_lm.item() + 0.1 * dev_loss_sft.item())

                    relative_improvement = (running_avg_performance - ( 0.9 * dev_loss_lm.item() + 0.1 * dev_loss_sft.item())) / running_avg_performance

                    current_weights = torch.tensor(train_dataloader.sampling_rates, dtype=torch.float64, device=fabric.device)
                    #X = torch.cat([X, current_weights.unsqueeze(0)], dim=0)
                    #Y = torch.cat([Y, torch.tensor([[relative_improvement]], dtype=torch.float64, device=fabric.device)]) 

                    moving_window_gp.add_observation(current_weights, relative_improvement)
                    X, Y = moving_window_gp.get_data()

                # TODO: make sure all Xs and Ys are gathered across GPUs
                # Generate new weights using Thompson Sampling
                if len(X) > 5:  # Ensure we have at least 5 data points for GP 
                    # if fabric.local_rank == 0:               
                    #     candidate_weights = generate_ts_candidates(X, Y, n_candidates=1000, batch_size=1)
                    #     new_sampling_rate = round_and_normalize(candidate_weights).tolist()[0]
                    # else:
                    #     new_sampling_rate = [0.0] * len(train_dataloader.loaders)
                    
                    # new_sampling_rate = fabric.broadcast(new_sampling_rate, src=0)

                    candidate_weights = generate_ts_candidates(X, Y, n_candidates=1000, batch_size=1)
                    new_sampling_rate = round_and_normalize(candidate_weights).tolist()[0]
                    new_sampling_rate = fabric.broadcast(new_sampling_rate, src=0)
                else:
                    # just pick a random candidate
                    #sobol = SobolEngine(dimension=n_datasets, scramble=True)
                    #new_sampling_rate = sobol.draw(1).tolist()[0]
                    # if fabric.local_rank == 0:
                    #     breakpoint()
                    #     new_sampling_rate = sample_from_simplex(1, X.shape[-1]).to(dtype=X.dtype, device=X.device).tolist()[0]
                    # else:
                    #     new_sampling_rate = [0.0] * len(train_dataloader.loaders)
                    
                    # new_sampling_rate = fabric.broadcast(new_sampling_rate, src=0)

                    new_sampling_rate = sample_from_simplex(1, X.shape[-1]).to(dtype=X.dtype, device=X.device).tolist()[0]
                    new_samplng_rate = fabric.broadcast(new_sampling_rate, src=0)

                #new_sampling_rate = train_dataloader.sampling_rates # TODO remove this
                
            # Update sampling rates in the dataloader
            fabric.print(f"Old sampling rates: {train_dataloader.sampling_rates}")
            train_dataloader.sampling_rates = (
                new_sampling_rate or train_dataloader.sampling_rates
            )
            train_dataloader._dataloader.sampling_rates = (
                new_sampling_rate or train_dataloader.sampling_rates
            )

            fabric.print(f"Updated sampling rates: {new_sampling_rate}")

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

        if train.lr_scheduler == "decay":
            lr = get_lr_decay_stage(
                initial_lr,
                state["iter_num"],
                initial_iter,
                train.min_lr,
                train.gradient_accumulation_iters(devices),
            )
        elif train.lr_scheduler == "cosine":
            # determine and set the learning rate for this iteration
            lr = get_lr(
                2e-5,  # the default LR is too high. Using this one
                state["iter_num"],
                warmup_iters,
                max_iters,
                train.min_lr,
            )
        elif train.lr_scheduler == "constant":
            lr = initial_lr

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

        # Process LM data
        is_accumulating = (
            state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        )

        all_losses = []

        lm_samples = 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            if lm_data is not None:
                input_ids = lm_data[:, 0 : model.max_seq_length].contiguous().long()
                targets = lm_data[:, 1 : (model.max_seq_length + 1)].contiguous().long()
                lm_loss = compute_lm_loss(model, input_ids, targets)

                all_losses.append(lm_loss)
                lm_loss_total.update(lm_loss.detach())
                lm_samples += input_ids.size(0)

                if use_adaptive_sampling and train.data_scheduler == "grad":
                    #model.zero_grad()
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

                    if use_adaptive_sampling and train.data_scheduler == "grad":
                        #model.zero_grad()
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
            sft_num_samples["lm"] = lm_samples

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
            num_samples_str = " | ".join(
                [f"num {name}: {num}" for name, num in sft_num_samples.items()]
            )
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" loss train (lm): {metrics['loss_lm']}",
                f" loss train (sft): {metrics['loss_sft']}",
                f" ~~ breakdown: {sft_sets_str} ~~"
                f" ~~ num samples: {num_samples_str} ~~"
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
