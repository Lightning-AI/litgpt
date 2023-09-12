import json
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lightning as L
from lit_gpt.lora import GPT as LoRAGPT
from lit_gpt.lora import merge_lora_weights as _merge_lora_weights
from lit_gpt.utils import lazy_load


def merge_lora_weights(
    checkpoint_path: Path,
    lora_path: Path,
    save_merge: bool = False,
) -> None:
    fabric = L.Fabric(devices=1)

    with open(lora_path.parent / "lit_lora_config.json", "r") as lora_config_path:
        config = json.load(lora_config_path)["config"]

    with fabric.init_module(empty_init=True):
        model = LoRAGPT.from_name(**config)

    # load the checkpoints. update the base checkpoint with lora weights. load the state_dict
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(lora_path) as lora_checkpoint:
        checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
        model.load_state_dict(checkpoint, strict=False)

    # merge the weights into the model
    _merge_lora_weights(model)

    # save the merged state_dict
    if save_merge:
        merged_path = checkpoint_path.parent / "lit_model_lora_merged.pth"
        fabric.print(f"Saving merged checkpoint to {merged_path}", file=sys.stderr)
        fabric.save(merged_path, {"model": model.state_dict()})
