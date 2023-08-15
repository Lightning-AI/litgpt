import json
import sys
import time
import warnings
from pathlib import Path
from typing import List, Literal, Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from lm_eval import base, evaluator, tasks
from lm_eval.base import BaseLM

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import Block
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load, quantization


class EvalHarnessBase(BaseLM):
    # Credits:
    # https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py
    def __init__(
        self,
        checkpoint_dir: str = "",
        precision: str = "bf16-true",
        batch_size=1,
        temperature=1.0,
        device="auto",
        devices: int = 1,
        strategy: str = "auto",
        quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    ):
        super().__init__()
        assert isinstance(device, str)
        assert isinstance(batch_size, int)
        assert isinstance(checkpoint_dir, str)
        self.batch_size_per_gpu = batch_size
        self.temperature = temperature

        if strategy == "fsdp":
            strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
        self.fabric = fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
        fabric.launch()

        checkpoint_dir = Path(checkpoint_dir)
        check_valid_checkpoint_dir(checkpoint_dir)

        with open(checkpoint_dir / "lit_config.json") as fp:
            config = Config(**json.load(fp))

        if quantize is not None and devices > 1:
            raise NotImplementedError
        if quantize == "gptq.int4":
            model_file = "lit_model_gptq.4bit.pth"
            if not (checkpoint_dir / model_file).is_file():
                raise ValueError("Please run `python quantize/gptq.py` first")
        else:
            model_file = "lit_model.pth"
        checkpoint_path = checkpoint_dir / model_file

        fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
        t0 = time.perf_counter()
        with fabric.init_module(empty_init=True), quantization(quantize):
            model = GPT(config)
        fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

        t0 = time.perf_counter()
        with lazy_load(checkpoint_path) as checkpoint:
            model.load_state_dict(checkpoint.get("model", checkpoint), strict=quantize is None)
        fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

        model.eval()
        self.model = fabric.setup_module(model)
        self.tokenizer = Tokenizer(checkpoint_dir)
        self.vocab_size = self.tokenizer.vocab_size

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        kwargs = {el.split("=")[0]: el.split("=")[1] for el in arg_string.split(",")}
        return cls(**kwargs, **additional_config)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_id

    @property
    def max_length(self):
        # TODO: keep decoupled from block_size
        return self.model.config.block_size

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu * self.fabric.world_size

    @property
    def device(self):
        return self.fabric.device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, bos=False, eos=False).tolist()

    def tok_decode(self, tokens):
        t = torch.tensor(tokens)
        return self.tokenizer.decode(t)

    @torch.no_grad()
    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        return self.model(inps)

    def _model_generate(self, context, max_length, eos_token_id):
        assert context.shape[0] == 1
        out = generate(
            model=self.model,
            idx=context[0],
            max_new_tokens=max_length,
            max_seq_length=self.model.config.block_size,
            temperature=self.temperature,
            top_k=None,
            eos_id=eos_token_id,
        )

        return self.tokenizer.decode(out)

    @torch.no_grad()
    def run_eval(
        self,
        eval_tasks=None,
        num_fewshot=0,
        bootstrap_iters=2,
        description_dict=None,
        use_cache=True,
        name="lit-gpt",
        limit=None,
    ):
        if eval_tasks is None:
            eval_tasks = ["arc_challenge", "piqa", "hellaswag", "hendrycksTest-*"]

        # Returns a list containing all values of the task registry that
        # match at least one of the patterns
        import fnmatch

        def pattern_match(patterns, source_list):
            task_names = set()
            for pattern in patterns:
                for matching in fnmatch.filter(source_list, pattern):
                    task_names.add(matching)
            return list(task_names)

        eval_tasks = pattern_match(eval_tasks, tasks.ALL_TASKS)
        print(f"Found tasks: {eval_tasks}")

        # **HACK INCOMING**:
        # first get task dict on local main rank
        # the tasks are downloaded *as they are initialized*, and the downloads don't like multithreading.
        # so we download them once on the local main rank, wait, and then initialize them on all other ranks, which *should* load from the cache.
        if self.fabric.local_rank == 0:
            tasks.get_task_dict(eval_tasks)
        # torch barrier
        self.fabric.barrier()
        tasks.get_task_dict(eval_tasks)

        lm = self
        if use_cache:
            lm = base.CachingLM(lm, "lm_cache/" + name + ".db")

        results = evaluator.evaluate(
            lm=lm,
            task_dict=tasks.get_task_dict(eval_tasks),
            description_dict=description_dict,
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
        )

        results["config"] = {
            "model": self.model.config.name,
            "num_fewshot": num_fewshot,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "no_cache": not use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "description_dict": description_dict,
        }

        return results


def run_eval_harness(
    checkpoint_dir: str = "",
    precision: Optional[str] = None,
    batch_size=1,
    eval_tasks: Optional[List[str]] = None,
    num_fewshot=0,
    bootstrap_iters=2,
    temperature=1.0,
    device="auto",
    devices: int = 1,
    strategy: str = "auto",
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    save_filepath: Optional[str] = None,
):
    precision = precision or get_default_supported_precision(training=False)

    eval_harness = EvalHarnessBase(
        checkpoint_dir=checkpoint_dir,
        precision=precision,
        batch_size=batch_size,
        temperature=temperature,
        device=device,
        devices=devices,
        strategy=strategy,
        quantize=quantize,
    )
    eval_harness.fabric.print("Running evaluation harness...")
    results = eval_harness.run_eval(
        eval_tasks=eval_tasks, num_fewshot=num_fewshot, bootstrap_iters=bootstrap_iters, use_cache=False
    )
    if save_filepath:
        data = json.dumps(results)
        with open(save_filepath, "w") as fw:
            fw.write(data)
        print(f"Results saved at {save_filepath}")
    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    result = CLI(run_eval_harness)
    print(result)
