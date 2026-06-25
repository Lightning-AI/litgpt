# Contributing to LitGPT

We welcome all contributions, regardless of your level of experience or hardware. Whether it's a bug fix, a new feature, or an improvement to the docs — we appreciate your help!

## How to contribute

1. **Open an issue** — describe the bug or feature before writing code. This helps us align on scope early.
2. **Fork the repo** and create a branch from `main`.
3. **Make your changes** and add or update relevant tests.
4. **Open a pull request** against `main`. Include a clear description of what changed and why.

## Development setup

```bash
git clone https://github.com/<your-username>/litgpt
cd litgpt
```

Install the package in editable mode with pip:

```bash
pip install -e ".[extra,test]"
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra extra --extra test
```

Install pre-commit hooks to catch style issues before pushing:

```bash
pip install pre-commit  # or: uv tool install pre-commit
pre-commit install
```

## Running tests

```bash
pytest tests/
```

## Guidelines

- Keep pull requests focused — one logical change per PR.
- Write tests for new functionality.
- Follow the existing code style (enforced via [ruff](https://docs.astral.sh/ruff/) and pre-commit).
- All code should be your own original work; third-party snippets must be attributed.

## Community

- [Request a feature or report a bug](https://github.com/Lightning-AI/litgpt/issues)
- [Contribution tutorial](https://lightning.ai/pages/community/tutorial/how-to-contribute-to-litgpt/)
- [Join our Discord](https://discord.gg/VptPCZkGNa)
