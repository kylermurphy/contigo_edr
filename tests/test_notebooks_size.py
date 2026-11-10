"""N3: keep the tutorial notebooks small and the repository root free of notebooks."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"
MAX_BYTES = 1024 * 1024  # 1 MB


def test_n3_tutorial_notebooks_under_1mb():
    """Every notebook in ``notebooks/tutorials`` is below 1 MB (clear large outputs)."""
    tutorials = NOTEBOOKS / "tutorials"
    if not tutorials.is_dir():
        pytest.skip("notebooks/tutorials not found in the repository (sdist?)")
    notebooks = sorted(tutorials.glob("*.ipynb"))
    assert notebooks, "no tutorial notebooks found"
    too_big = {
        nb.name: nb.stat().st_size for nb in notebooks if nb.stat().st_size >= MAX_BYTES
    }
    assert too_big == {}


def test_n3_no_notebooks_in_repo_root():
    """Notebooks live under ``notebooks/``; none is left in the repository root."""
    if not NOTEBOOKS.is_dir():
        pytest.skip("notebooks folder not found in the repository (sdist?)")
    assert sorted(p.name for p in REPO.glob("*.ipynb")) == []
