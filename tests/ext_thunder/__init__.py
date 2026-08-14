import sys
from pathlib import Path

# support running without installing as a package, adding extensions to the Python path
wd = Path(__file__).parent.parent.parent.resolve()
if wd.is_dir():
    sys.path.append(str(wd))
else:
    import warnings

    warnings.warn(f"Could not find extensions directory at {wd}")
