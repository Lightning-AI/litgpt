# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Maintainer-only: publishes gated HF tokenizer/config fixtures to the Lightning Model Registry."""

import argparse
from pathlib import Path

import litdata
from _fixtures import GATED_TOKENIZER_REPOS, TOKENIZER_FILES, fixture_name, fixture_slug
from huggingface_hub import snapshot_download


def publish_fixture(repo_id: str, output_dir: str) -> None:
    """Downloads `repo_id`'s tokenizer/config files and publishes them as a registry model."""
    from litmodels import upload_model

    fixture_dir = Path(output_dir) / f"{fixture_slug(repo_id)}-tokenizer"
    snapshot_download(repo_id, local_dir=fixture_dir, allow_patterns=list(TOKENIZER_FILES))
    upload_model(name=fixture_name(repo_id), model=str(fixture_dir), progress_bar=False, verbose=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repos",
        nargs="*",
        default=list(GATED_TOKENIZER_REPOS),
        help="HF repo ids to publish (default: all gated repos in the fixture map).",
    )
    parser.add_argument(
        "--staging-dir",
        default="litgpt-ci-fixtures",
        help="Local directory to stage downloaded files before upload.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="How many repos to publish in parallel (downloads/uploads are I/O bound).",
    )
    args = parser.parse_args()

    unknown = [r for r in args.repos if r not in GATED_TOKENIZER_REPOS]
    if unknown:
        parser.error(f"Not in the gated fixture map: {unknown}")

    litdata.map(
        fn=publish_fixture,
        inputs=list(args.repos),
        output_dir=args.staging_dir,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
