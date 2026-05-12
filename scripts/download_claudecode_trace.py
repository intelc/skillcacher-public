"""Stage the public ClaudeCodeTrace HF dataset locally for replay.

`huggingface_hub.snapshot_download` is the canonical fetch path — the
HF Dataset Viewer doesn't auto-render the nested-dir layout, so the
CLI/SDK is the way in. HF_TOKEN is honored if set; read-scope is
sufficient for a public dataset."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import click

REPO_ID = "intelchen/claudecode-trace"
EXPECTED_SUBSETS = {"swebench_verified", "skill_invocation", "post_compact"}


def stage_dataset(
    local_dir: Path,
    *,
    revision: str | None = None,
    token: str | None = None,
    snapshot_download_fn=None,
) -> Path:
    """Download the dataset to local_dir. Returns the resolved staged path.

    `snapshot_download_fn` is injectable for tests; production calls
    `huggingface_hub.snapshot_download`."""
    if snapshot_download_fn is None:
        try:
            from huggingface_hub import snapshot_download as snapshot_download_fn
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub not installed; pip install huggingface_hub"
            ) from e

    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download_fn(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
        revision=revision,
        token=token,
    )
    return Path(path)


def validate_subsets(staged: Path) -> set[str]:
    """Return the set of expected subsets that are missing from staged."""
    if not staged.exists():
        return set(EXPECTED_SUBSETS)
    present = {p.name for p in staged.iterdir() if p.is_dir()}
    return EXPECTED_SUBSETS - present


@click.command()
@click.option("--local-dir", default="datasets/claudecode-trace",
              type=click.Path(path_type=Path),
              help="Where to stage the dataset locally")
@click.option("--revision", default=None,
              help="Optional revision (branch / commit). Defaults to main.")
def main(local_dir: Path, revision: str | None) -> None:
    """Download the ClaudeCodeTrace HF dataset to LOCAL_DIR."""
    token = os.environ.get("HF_TOKEN") or None
    try:
        staged = stage_dataset(local_dir, revision=revision, token=token)
    except RuntimeError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    missing = validate_subsets(staged)
    if missing:
        click.echo(
            f"WARN: expected subsets missing from staged dataset: {sorted(missing)}",
            err=True,
        )
    click.echo(str(staged))


if __name__ == "__main__":
    main()
