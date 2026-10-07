from pathlib import Path

import pytest

from ards_ai.cli.train_tabular import ensure_outside_repository

REPOSITORY = Path(__file__).resolve().parents[1]


def test_output_inside_repository_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the public Git repository"):
        ensure_outside_repository(REPOSITORY / "outputs" / "unsafe-run")


def test_output_outside_repository_is_allowed(tmp_path: Path) -> None:
    ensure_outside_repository(tmp_path / "safe-run")
