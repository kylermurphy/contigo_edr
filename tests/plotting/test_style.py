"""Tests for contigo.plotting._style (PL-09 to PL-11).

added: 09/11/2026 Claude Code
"""

import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from contigo.plotting import _style  # noqa: E402


def test_pl09_color_for_is_deterministic() -> None:
    """PL-09: an ID maps to a fixed tab20 entry (crc32, not hash())."""
    c1 = _style.color_for("A")
    assert c1 == _style.color_for("A")
    assert len(c1) == 4 and all(0.0 <= v <= 1.0 for v in c1)
    # str() based: numeric and string forms of an ID agree
    assert _style.color_for(25544) == _style.color_for("25544")
    # fixed expectation: independent of PYTHONHASHSEED and of the process
    expected_bin = zlib.crc32(b"CHAMP") % 20
    assert expected_bin == 17
    assert _style.color_for("CHAMP") == tuple(
        float(v) for v in matplotlib.colormaps["tab20"](expected_bin)
    )


def test_pl09b_color_map_resolves_clashes() -> None:
    """PL-09b: IDs sharing a crc32 bin still get distinct colours in one plot,
    the result does not depend on input order, and non-clashing IDs keep
    their color_for colour."""
    # 'CHAMP' and 'SWARM-A' both fall in bin 17
    assert zlib.crc32(b"SWARM-A") % 20 == zlib.crc32(b"CHAMP") % 20
    ids = ["SWARM-A", "CHAMP", "GRACE"]
    cmap = _style.color_map(ids)
    assert list(cmap) == ids
    assert len(set(cmap.values())) == len(ids)
    assert cmap == _style.color_map(list(reversed(ids)))
    assert cmap["CHAMP"] == _style.color_for("CHAMP")  # first in sorted order
    assert cmap["GRACE"] == _style.color_for("GRACE")


def test_pl10_rc_context_leaves_no_global_state() -> None:
    """PL-10: rcParams are unchanged after leaving the context."""
    plt = _style.require_matplotlib()
    before = dict(matplotlib.rcParams)
    with _style.rc_context():
        assert plt.rcParams["axes.grid"] is True
    assert dict(matplotlib.rcParams) == before


_NO_MPL_SCRIPT = """
import sys
sys.modules["matplotlib"] = None          # simulate matplotlib not installed
sys.modules["matplotlib.pyplot"] = None
import contigo.plotting, contigo.plotting._util, contigo.plotting._style as s
assert contigo.plotting.__all__ == []
for fn in (s.require_matplotlib, lambda: s.color_for("A")):
    try:
        fn()
    except ImportError as err:
        assert "contigo[plot]" in str(err), err
        assert isinstance(err.__cause__, ImportError)
    else:
        raise AssertionError("expected ImportError")
print("ok")
"""


def test_pl11_package_imports_without_matplotlib(tmp_path) -> None:
    """PL-11: in a fresh interpreter with matplotlib blocked, contigo.plotting
    imports and the helpers raise an ImportError pointing to contigo[plot].

    Runs in a subprocess (modules already imported in this process would make
    the check a no-op), with cwd outside the repo and ``PYTHONPATH`` set to the
    copy of contigo under test, so the subprocess imports the same package as
    this test session (an editable install may point at another checkout).
    """
    import contigo

    pkg_root = str(Path(contigo.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": pkg_root}
    res = subprocess.run(
        [sys.executable, "-c", _NO_MPL_SCRIPT + "print(contigo.__file__)\n"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout.split()
    assert out[0] == "ok"
    assert Path(out[1]).resolve() == Path(contigo.__file__).resolve()
