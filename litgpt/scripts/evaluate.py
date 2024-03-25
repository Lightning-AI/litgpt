# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
from pathlib import Path
from typing import Optional
import torch

from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import CLI, copy_config_files


def safe_safetensors(out_dir, repo_id):
    from transformers import AutoModel

    state_dict = torch.load(out_dir/"model.pth")
    model = AutoModel.from_pretrained(
        repo_id, state_dict=state_dict
    )
    model.save_pretrained(out_dir)


def prepare_results(results, save_filepath, print_results=True):
    from lm_eval.utils import make_table

    if print_results:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

    json_result = json.dumps(
        results, indent=2, ensure_ascii=False
    )
    save_filepath.open("w", encoding="utf-8").write(json_result)


def convert_and_evaluate(
    checkpoint_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    repo_id: Optional[str] = None,
    skip_conversion: bool = False,
    tasks: Optional[str] = "hellaswag,gsm8k,truthfulqa_mc2,mmlu,winogrande,arc_challenge",
    num_fewshot: Optional[int] = None,
    batch_size: int = 1,
    device: Optional[str] = None,
    limit: Optional[float] = None,
    seed: int = 1234,
    save_filepath: Optional[str] = None,
) -> None:
    """Convert a LitGPT model and run the LM Evaluation Harness

    Arguments:
        checkpoint_dir: Directory where the `lit_model.pth` and tokenizer files are located.
        out_dir: Directory in which to save the converted checkpoints for evaluation.
        repo_id: The original repo ID the model was derived from.
        skip_conversion: Set to `True` to skip the model conversion,
            assuming the model has already been converted and the
            model.pth and .safetensor files exist.
        tasks: CSV of task names to evaluate.
           By default, the Open LM Leaderboard tasks are used:
           "hellaswag,gsm8k,truthfulqa_mc2,mmlu,winogrande,arc_challenge"
        num_fewshot: Number of examples in few-shot context.
        batch_size: Batch size configuration.
        device: Device to use for evaluation, for example, "cuda" or "cuda:0".
        limit: Limit on number of examples per task.
        seed: Random seed.
        save_filepath: The file where the results will be saved. 
            Saves to `out_dir/results.json` by default.
    """

    from lm_eval import evaluator

    if checkpoint_dir is None:
        raise ValueError("Provide a checkpoint_dir argument.")
    if out_dir is None:
        raise ValueError("Provide a checkpoint_dir argument.")
    if repo_id is None:
        raise ValueError("Provide a repo_id argument.")   

    checkpoint_dir, out_dir = Path(checkpoint_dir), Path(out_dir)

    save_filepath = out_dir / Path("results.json") if save_filepath is None else Path(save_filepath)

    out_dir.mkdir(parents=True, exist_ok=True)

    copy_config_files(source_dir=checkpoint_dir, out_dir=out_dir)

    if not skip_conversion:
        convert_lit_checkpoint(checkpoint_dir=checkpoint_dir, output_dir=out_dir)
        safe_safetensors(out_dir, repo_id)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={out_dir}",
        tasks=tasks.split(","),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )

    print("results", results)
    prepare_results(results, save_filepath)


if __name__ == "__main__":
    CLI(convert_and_evaluate)
