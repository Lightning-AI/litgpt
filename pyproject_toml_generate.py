import tomli, tomli_w
from pathlib import Path

root = Path(__file__).parent
requirements_path = Path(root / 'requirements-all.txt')
pyproject_path = Path(root / 'pyproject.toml')

requirements_all = (requirements_path).read_text().split("\n")
requirements_all = [
    r.split("#")[0].strip()
    for r in requirements_all if r and not r.strip().startswith("-r")
]

with pyproject_path.open('rb') as f:
    pyproject_data = tomli.load(f)

pyproject_data.setdefault('project', {}).setdefault('optional-dependencies', {})['all'] = requirements_all

with pyproject_path.open('wb') as f:
    tomli_w.dump(pyproject_data, f)