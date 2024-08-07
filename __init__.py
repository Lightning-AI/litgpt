# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import importlib
import os
import warnings

# Check if litgpt is installed as a package and exists as directory with the same name in the current working directory
package_installed = importlib.util.find_spec("litgpt") is not None
directory_exists = os.path.isdir("litgpt")
if package_installed and directory_exists:
    warnings.warn(
        "The package 'litgpt' is installed and a directory with the same name exists in the working directory. "
        "Please rename the 'litgpt' directory or move it to a subdirectory to avoid import conflicts.", UserWarning)
