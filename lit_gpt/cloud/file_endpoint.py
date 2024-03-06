
import requests
from time import sleep
import os
from typing import Dict, List, Optional


class FileEndpoint:
    
    def __init__(
        self,
        api_key: str,
        url: str,
        args: Optional[Dict[str, str]] = {},
        files: Dict[str, str] = {}
    ):
        """
        This class enables to connect to a file endpoint on the Lightning AI Platform.

        Arguments:
            api_key: The API Key of the user
            url: The File Endpoint URL
            args: The provided arguments
            files: The provided files
        """
        self.api_key = api_key
        self.url = url
        self.init_args = args
        self.init_files = files

    def run(
        self,
        args: Dict[str, str] = {},
        files: Dict[str, str] = {},
        output_dir: str = "results"
    ):
        os.makedirs(output_dir, exist_ok=True)

        args = {**self.init_args, **args}
        files = {**self.init_files, **files}

        response = requests.post(self.url, json=args)

        if response.status_code != 200:
            raise Exception(f"The file endpoint isn't reachable. Found status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print(data)

            if "files_to_upload" in data:
                files_to_upload = data["files_to_upload"]

                if len(files) != len(files_to_upload):
                    raise ValueError(f"This endpoint is expecting {len(files_to_upload)} files to be uploaded. Found {files}.")

                for file_to_upload in files_to_upload:
                    upload_id = file_to_upload['upload_id']
                    name = file_to_upload['name']
                    url = f"{self.url}?upload_id={upload_id}"
                    response = requests.post(url, files={upload_id: open(files[name], "rb")})

            while True:
                url = f"{self.url}?run_id={data['run_id']}"
                response = requests.post(url)

                sleep(1)

                if response.status_code != 200:
                    continue

                data = response.json()
                print(data)

                if data["stage"] == "completed":
                    break


                if data["stage"] == "failed":
                    print("The Studio File Endpoint failed")
                    import sys
                    sys.exit(0)

            if 'download_ids' in data:
                for download_id in data['download_ids']:
                    url = f"{self.url}?download_id={download_id}"

                    with requests.post(url, stream=True) as r:
                        r.raise_for_status()
                        filename = r.headers["Content-Disposition"].split("filename=")[1]
                        filename = os.path.basename(filename)
                        with open(os.path.join(output_dir, filename), 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192): 
                                f.write(chunk)