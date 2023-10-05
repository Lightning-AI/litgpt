import json
import sys
from pathlib import Path
from typing import List, Literal, Optional

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lm_eval import base, evaluator, tasks
from lm_eval.base import BaseLM

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    gptq_quantization,
    load_checkpoint,
)


class EvalHarnessBase(BaseLM):
    # Credits:
    # https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py
    def __init__(self, fabric: L.Fabric, model: GPT, tokenizer: Tokenizer, batch_size: int, temperature: float):
        super().__init__()
        self.fabric = fabric
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size_per_gpu = batch_size
        self.temperature = temperature

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
        return self.model.max_seq_length

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

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

    @torch.inference_mode()
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
            self.model, context[0], max_length, temperature=self.temperature, top_k=None, eos_id=eos_token_id
        )

        return self.tokenizer.decode(out)

    @torch.inference_mode()
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


@torch.inference_mode()
def run_eval_harness(
    checkpoint_dir: Path,
    precision: Optional[str] = None,
    batch_size=1,
    temperature=1.0,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    eval_tasks: Optional[List[str]] = None,
    num_fewshot=0,
    bootstrap_iters=2,
    save_filepath: Optional[Path] = None,
):
    if precision is None:
        precision = get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    checkpoint_dir = Path(checkpoint_dir)
    check_valid_checkpoint_dir(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)

    model.eval()
    model = fabric.setup_module(model)

    load_checkpoint(fabric, model, checkpoint_path)

    eval_harness = EvalHarnessBase(fabric, model, tokenizer, batch_size, temperature)

    results = eval_harness.run_eval(
        eval_tasks=eval_tasks, num_fewshot=num_fewshot, bootstrap_iters=bootstrap_iters, use_cache=False
    )
    if save_filepath is None:
        print(results)
    else:
        print(f"Saving results to {str(save_filepath)!r}")
        data = json.dumps(results)
        with open(save_filepath, "w") as fw:
            fw.write(data)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(run_eval_harness, as_positional=False)
