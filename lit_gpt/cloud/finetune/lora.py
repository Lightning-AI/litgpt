from lit_gpt.cloud.file_endpoint import FileEndpoint
from typing import Dict
import os

class FinetuneLora(FileEndpoint):

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            url="https://finetune-lora-01hr9fb4nxd6dc4j08cs2bgbyz.cloudspaces.litng.ai",
            args={"data": "JSON", "train.max_steps": "100"},
            files={"config": "config_hub/lora/lora-finetune.yaml"},
        )

    def run(self, data_path: str, output_dir: str = "results", **kwargs):
        args, files = {}, {}
        files["data.json_path"] = data_path

        for k, v in kwargs.items():
            if os.path.exists(v):
                files[k] = v
            else:
                args[k] = v

        super().run(args, files, output_dir)