# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import sys
import json
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.utils import CLI


def upgrade(root_dir: Path = Path("."), write: bool = False) -> False:
    config_files = root_dir.rglob("*lit_config.json")
    files_to_update = []

    for filepath in config_files:
        with open(filepath, "r") as file:
            config = json.load(file)
        updated_config = config.clone()

        if "name" in config and "org" in config and not "hf_config" in config:
            name = updated_config["name"]
            org = updated_config.pop("org")
            updated_config["hf_config"] = {"name": name, "org": org}
        if "_norm_class" in config:
            updated_config["norm_class_name"] = config.pop("_norm_class")
        if "_mlp_class" in config:
            updated_config["mlp_class_name"] = config.pop("_mlp_class")
        if "condense_ratio" in config:
            updated_config["rope_condense_ratio"] = config.pop("condense_ratio")

        if write:
            with open(filepath, "w") as file:
                json.dump(updated_config, file)
        elif updated_config != config:
            files_to_update.append(filepath)

    if files_to_update:
        print("The following files will be updated:\n")
        for file in files_to_update:
            print(file)
        print("\nRerun the previous command with `--write` to write the changes to all these files.")


if __name__ == "__main__":
    CLI(upgrade)
