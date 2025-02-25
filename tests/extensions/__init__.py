import sys
from pathlib import Path

# support running without installing as a package, adding extensions to the Pyton path
wd = Path(__file__).parent.parent.resolve() / "extensions"
sys.path.append(str(wd))