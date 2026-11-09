"""Tests for contigo.plotting._style (PL-09 to PL-11).

added: 09/11/2026 Claude Code
"""

import os
import subprocess
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

from contigo.plotting import _style  # noqa: E402


def test_pl09_color_for_is_deterministic() -> None:
    """PL-09: same ID, same colour, across calls and across processes."""
    c1 = _style.color_for("A")
    assert c1 == _style.color_for("A")
    assert len(c1) == 4 and all(0.0 <= v <= 1.0 for v in c1)
    # str() based: numeric and string forms of an ID agree
    assert _style.color_for(25544) == _style.color_for("25544")

    code = (
        "import matplotlib; matplotlib.use('Agg');"
        "from contigo.plotting import _style;"
        "print(_style.color_for('CHAMP'))"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("1", "2")
    }
    assert len(outs) == 1
    assert outs == {str(_style.color_for("CHAMP"))}


def test_pl10_rc_context_leaves_no_global_state() -> None:
    """PL-10: rcParams are unchanged after leaving the context."""
    plt = _style.require_matplotlib()
    before = dict(matplotlib.rcParams)
    with _style.rc_context():
        assert plt.rcParams["axes.grid"] is True
    assert dict(matplotlib.rcParams) == before


def test_pl11_package_imports_without_matplotlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PL-11: contigo.plotting imports without matplotlib; require_matplotlib
    raises an ImportError that points to contigo[plot]."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    import contigo.plotting
    import contigo.plotting._util  # noqa: F401

    assert contigo.plotting.__all__ == []
    with pytest.raises(ImportError, match=r"contigo\[plot\]") as exc:
        _style.require_matplotlib()
    assert isinstance(exc.value.__cause__, ImportError)
