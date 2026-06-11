# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Maintainer-only: mirror gated HF tokenizer/config files to the Lightning Model Registry.

This is NOT a test (no ``test_`` prefix, so pytest will not collect it). Run it from a
trusted environment that has both a valid ``HF_TOKEN`` (with access to the gated repos)
and a Lightning login (``lightning login`` / ``LIGHTNING_API_KEY``):

    HF_TOKEN=... python tests/upload_gated_tokenizer_fixtures.py            # all mapped repos
    HF_TOKEN=... python tests/upload_gated_tokenizer_fixtures.py meta-llama/Llama-2-7b-hf

It downloads only tokenizer/config files (never weights), records a small provenance
file, and uploads each as a version-pinned model so fork-PR CI can fetch it without any
secret. The registry naming and repo list are shared with ``tests/_fixtures.py`` so the
upload targets always match what the tests resolve.

Note: the destination teamspace (``lightning-ai/litgpt-ci``) must exist and its models
must be made public (a one-time maintainer/admin action in the Lightning UI) for the
anonymous guest-login download path in CI to work.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from _fixtures import GATED_TOKENIZER_REPOS, TOKENIZER_FILES, fixture_name, fixture_slug
from huggingface_hub import snapshot_download


def upload_fixture(repo_id: str, staging_root: Path) -> str:
    """Download tokenizer/config files for ``repo_id`` and upload them as a registry model."""
    from litmodels import upload_model

    fixture_dir = staging_root / f"{fixture_slug(repo_id)}-tokenizer"
    snapshot_download(
        repo_id,
        local_dir=fixture_dir,
        allow_patterns=list(TOKENIZER_FILES),
    )

    (fixture_dir / "LITGPT_FIXTURE_SOURCE.txt").write_text(
        f"source_repo={repo_id}\ncreated_at={datetime.now(timezone.utc).isoformat()}\nfiles=tokenizer/config only\n",
        encoding="utf-8",
    )

    registry_name = fixture_name(repo_id)  # includes the pinned :version
    upload_model(name=registry_name, path=str(fixture_dir))
    return registry_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repos",
        nargs="*",
        default=list(GATED_TOKENIZER_REPOS),
        help="HF repo ids to mirror (default: all gated repos in the fixture map).",
    )
    parser.add_argument(
        "--staging-dir",
        default="litgpt-ci-fixtures",
        help="Local directory to stage downloaded files before upload.",
    )
    args = parser.parse_args()

    unknown = [r for r in args.repos if r not in GATED_TOKENIZER_REPOS]
    if unknown:
        parser.error(f"Not in the gated fixture map: {unknown}")

    staging_root = Path(args.staging_dir)
    staging_root.mkdir(parents=True, exist_ok=True)

    for repo_id in args.repos:
        registry_name = upload_fixture(repo_id, staging_root)
        print(f"{repo_id} -> {registry_name}", file=sys.stderr)


if __name__ == "__main__":
    main()
