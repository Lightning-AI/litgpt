# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
import subprocess
import sys
from pathlib import Path
from pprint import pprint
from typing import Optional, Union

import torch

from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files, extend_checkpoint_dir


def prepare_results(results, save_filepath, print_results=True):
    from lm_eval.utils import make_table

    # dump samples to file
    # if "samples" in results:
    #     samples_filepath = save_filepath.parent / "samples.json"
    #     with samples_filepath.open("w") as f:
    #         for sample in results["samples"]:
    #             f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    #     print(f"Samples written to {samples_filepath}")

    if print_results:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))
    try:
        json_result = json.dumps(results["results"], indent=2, ensure_ascii=False)
        save_filepath.open("w", encoding="utf-8").write(json_result)
        print(f"Result saved at {save_filepath}")
    except:
        # TODO: troubleshoot this later
        print("Failed to save results, continuing...")


def print_results_to_file(results_str, save_path):
    with open(save_path, "w") as f:
        f.write(results_str)

    print(f"Eval harness result written to {save_path}")


def convert_and_evaluate(
    checkpoint_dir: Path,
    tasks: Optional[str] = None,
    out_dir: Optional[Path] = None,
    force_conversion: bool = True,
    num_fewshot: Optional[int] = None,
    batch_size: Union[int, str] = 1,
    device: Optional[str] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    limit: Optional[float] = None,
    seed: int = 1234,
    save_filepath: Optional[Path] = None,
    use_cli: bool = True,
    include_path: Optional[str] = None,
    parallelize: bool = False,
) -> None:
    """Evaluate a model with the LM Evaluation Harness.

    Arguments:
        checkpoint_dir: Directory where the `lit_model.pth` and tokenizer files are located.
        out_dir: Directory in which to save the converted checkpoints for evaluation.
            Saves to `checkpoint_dir`/evaluate by default.
        force_conversion: Set to `True` to reconvert the model and override
            an existing model.pth from a previous evaluation call.
        tasks: CSV of task names to evaluate. Example: "hellaswag,truthfulqa_mc2,mmlu"
        num_fewshot: Number of examples in few-shot context.
        batch_size: Batch size configuration as positive integer value (default: 1),
            "auto", in the format 'auto:N', where 'auto:4' recomputes the batch size 4 times.
        device: Device to use for evaluation, for example, "cuda" or "cuda:0".
        limit: Limit on number of examples per task.
        seed: Random seed.
        save_filepath: The file where the results will be saved.
            Saves to `out_dir/results.json` by default.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    if not (isinstance(batch_size, int) and batch_size > 0) and not (
        isinstance(batch_size, str) and batch_size.startswith("auto")
    ):
        raise ValueError(
            "batch_size must be a positive integer, 'auto', or in the format 'auto:N'."
        )

    from lm_eval import evaluator

    if tasks is None:
        from lm_eval.tasks import TaskManager

        taskm = TaskManager()
        print("\n".join(taskm.task_index.keys()))
        print(
            "\n\nTo evaluate multiple tasks, you can chain the task names "
            "listed above via a comma-separated list."
            "\nFor example: `--tasks 'hellaswag,truthfulqa_mc2,mmlu'`. "
            "\nTo search for a specific task, use `litgpt evaluate | grep task_name`."
        )
        return

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if parallelize and not use_cli:
        print(
            "Warning: parallelize is only supported with the lm_eval CLI. Ignoring parallelize."
        )

    if out_dir is None:
        out_dir = checkpoint_dir / "evaluate"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_filepath = (
        out_dir / Path("results.json") if save_filepath is None else Path(save_filepath)
    )

    model_path = out_dir / "pytorch_model.bin"
    if not model_path.exists() or force_conversion:
        copy_config_files(source_dir=checkpoint_dir, out_dir=out_dir)
        convert_lit_checkpoint(checkpoint_dir=checkpoint_dir, output_dir=out_dir)

        # Hack: LitGPT's conversion doesn't save a pickle file that is compatible to be loaded with
        # `torch.load(..., weights_only=True)`, which is a requirement in HFLM.
        # So we're `torch.load`-ing and `torch.sav`-ing it again to work around this.
        state_dict = torch.load(out_dir / "model.pth")
        torch.save(state_dict, model_path)
        os.remove(out_dir / "model.pth")

    from lm_eval.models.huggingface import HFLM

    model = HFLM(
        pretrained=str(out_dir.resolve()),
        device=device,
        batch_size=batch_size,
        dtype=dtype,
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if use_cli:
        model_args = f"pretrained={out_dir}"
        if parallelize:
            print("Parallelizing")
            model_args += ",parallelize=True"
        if include_path:
            results = subprocess.run(
                [
                    "with-proxy",
                    "lm_eval",
                    "--model",
                    "hf",
                    "--model_args",
                    model_args,
                    "--include_path",
                    include_path,
                    "--tasks",
                    tasks,
                    "--trust_remote_code",
                    "--num_fewshot",
                    str(num_fewshot),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ).stdout
        else:
            process = subprocess.Popen(
                [
                    "with-proxy",
                    "lm_eval",
                    "--model",
                    "hf",
                    "--model_args",
                    model_args,
                    "--tasks",
                    tasks,
                    "--trust_remote_code",
                    "--output_path",
                    save_filepath,
                    "--log_samples",
                    "--num_fewshot",
                    str(num_fewshot),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            results = []
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                results.append(line)

            process.wait()
            results = "".join(results)

        txt_save_filepath = str(save_filepath).replace(".json", ".txt")
        print_results_to_file(results, txt_save_filepath)
    else:
        results = evaluator.simple_evaluate(
            model=model,
            tasks=tasks.split(","),
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
        )
        prepare_results(results, save_filepath)
