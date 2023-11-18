import subprocess
import sys
from pathlib import Path

from conftest import RunIf
from lightning import Fabric


@RunIf(min_python="3.9")
def test_run_eval(tmp_path, float_like):
    from eval.lm_eval_harness import EvalHarnessBase
    from lit_gpt.model import GPT
    from lit_gpt.tokenizer import Tokenizer
    from scripts.download import download_from_hub

    fabric = Fabric(devices=1, precision="16-true")
    with fabric.init_module():
        model = GPT.from_name("pythia-70m")
    download_from_hub(repo_id="EleutherAI/pythia-70m", tokenizer_only=True, checkpoint_dir=tmp_path)
    tokenizer = Tokenizer(tmp_path / "EleutherAI/pythia-70m")

    eval_harness = EvalHarnessBase(fabric, model, tokenizer, 1)
    results = eval_harness.run_eval(
        eval_tasks=["truthfulqa_mc", "hellaswag", "hendrycksTest-machine_learning"],
        limit=2,
        bootstrap_iters=2,
        num_fewshot=0,
        no_cache=True,
    )
    assert results == {
        "config": {
            "batch_size": 1,
            "bootstrap_iters": 2,
            "device": "cuda:0",
            "limit": 2,
            "model": "pythia-70m",
            "no_cache": True,
            "num_fewshot": 0,
        },
        "results": {
            "hellaswag": {
                "acc": float_like,
                "acc_norm": float_like,
                "acc_norm_stderr": float_like,
                "acc_stderr": float_like,
            },
            "hendrycksTest-machine_learning": {
                "acc": float_like,
                "acc_norm": float_like,
                "acc_norm_stderr": float_like,
                "acc_stderr": float_like,
            },
            "truthfulqa_mc": {"mc1": float_like, "mc1_stderr": float_like, "mc2": float_like, "mc2_stderr": float_like},
        },
        "versions": {"hellaswag": 0, "hendrycksTest-machine_learning": 1, "truthfulqa_mc": 1},
    }


@RunIf(min_python="3.9")
def test_cli():
    cli_path = Path(__file__).parent.parent / "eval" / "lm_eval_harness.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "run_eval_harness" in output
