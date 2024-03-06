import os
import json
from typing import Dict
from pathlib import Path

RUN_PATH = os.getenv("RUN_PATH", None)

def replace_out_dir(out_dir: Path) -> Path:
    if RUN_PATH is None:
        return out_dir
    out_dir = Path(RUN_PATH) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _write(filename: str, progress_status: Dict[str, str]) -> None:
    if RUN_PATH is None:
        return
    
    with open(os.path.join(RUN_PATH, filename), "w") as f:
        json.dump(progress_status, f)

def write_progress_status(progress_status: Dict[str, str])-> None:
    _write("status.json", progress_status)

def write_success_status(success_status: Dict[str, str]) -> None:
    _write("success.json", success_status)

def write_failed_status(failed_status: Dict[str, str]) -> None:
    _write("failed.json", failed_status)