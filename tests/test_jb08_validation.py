"""Input validation of ``jb2008_density`` happens before the JVM starts (R3).

Every invalid input must raise ``ValueError`` without starting Java:
``jpype.startJVM`` is replaced by a guard that raises, so a validation path
that reached the JVM would fail with ``RuntimeError`` instead. A positive
control checks that valid input does reach the (stubbed) Orekit start, so the
ordering is really tested. Fast tier: no Java, no network.

Expected values: ``docs/phase0_plan.md`` F24 and R3; the validation rules in
``contigo/contigo_utils/jb08_utils.py`` (``_validate_geodetic`` and the
checks at the top of ``jb2008_density``).

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

from datetime import datetime, timedelta

import jpype
import numpy as np
import pytest

import contigo.config as config
from contigo.contigo_utils import jb08_utils, orekit_utils
from contigo.contigo_utils.jb08_utils import _SW_NAMES, jb2008_density

N = 3
TIMES = [datetime(2018, 1, 2) + timedelta(seconds=10 * i) for i in range(N)]
ECEF = np.array([[6.8e6, 0.0, 0.0], [0.0, 6.8e6, 0.0], [0.0, 0.0, 6.8e6]])
GEO = np.array([[10.0, 20.0, 400e3], [11.0, 21.0, 401e3], [12.0, 22.0, 402e3]])


class _StartAttempted(Exception):
    """Raised by the stubbed ``start_orekit`` in the positive control."""


@pytest.fixture(autouse=True)
def _no_jvm(monkeypatch):
    """Make any JVM start fail loudly and force the lazy start path."""
    if jpype.isJVMStarted():
        pytest.skip("JVM already running in this process")

    def _guard(*args, **kwargs):
        raise RuntimeError("JVM start attempted during input validation")

    monkeypatch.setattr(jpype, "startJVM", _guard)
    monkeypatch.setitem(config.state, "orekit_loaded", False)
    jb08_utils._java.cache_clear()
    yield
    jb08_utils._java.cache_clear()


def _sw(n: int = N, skip: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    return {name: np.full(n, 100.0) for name in _SW_NAMES if name not in skip}


@pytest.mark.parametrize(
    "positions, kwargs, match",
    [
        pytest.param(ECEF, {"coords": "eci"}, "coords must be one of", id="coords"),
        pytest.param(ECEF, _sw(skip=("dstdtc",)), "Missing", id="partial-sw-1"),
        pytest.param(ECEF, {"f10": np.ones(N)}, "Missing", id="partial-sw-8"),
        pytest.param(ECEF[:2], {}, r"shape \(N, 3\)", id="too-few-rows"),
        pytest.param(ECEF[:, :2], {}, r"shape \(N, 3\)", id="two-columns"),
        pytest.param(ECEF.ravel(), {}, r"shape \(N, 3\)", id="flat"),
        pytest.param(
            GEO * [9.5, 1, 1], {"coords": "geodetic"}, "latitude", id="lat>90"
        ),
        pytest.param(
            GEO - [0, 400, 0], {"coords": "geodetic"}, "longitude", id="lon<-180"
        ),
        pytest.param(GEO / 1000, {"coords": "geodetic"}, "in km", id="alt-km"),
        pytest.param(
            np.column_stack([GEO[:, :2], ECEF[:, 0]]),
            {"coords": "geodetic"},
            "ECEF Cartesian",
            id="alt-ecef",
        ),
        pytest.param(
            GEO - [0, 0, 350e3],
            {"coords": "geodetic"},
            "below the JB2008 minimum",
            id="alt-below-min",
        ),
    ],
)
def test_invalid_input_raises_before_jvm(positions, kwargs, match):
    """Invalid input raises ``ValueError``; no JVM start is attempted."""
    with pytest.raises(ValueError, match=match):
        jb2008_density(positions, TIMES, **kwargs)
    assert not jpype.isJVMStarted()


@pytest.mark.parametrize(
    "positions, kwargs",
    [
        pytest.param(ECEF, {}, id="ecef"),
        pytest.param(GEO, {"coords": "geodetic"}, id="geodetic"),
        pytest.param(ECEF, _sw(), id="ecef-user-sw"),
    ],
)
def test_valid_input_reaches_orekit_start(monkeypatch, positions, kwargs):
    """Positive control: valid input gets past validation to the Orekit start."""

    def _start(*args, **kwargs):
        raise _StartAttempted

    monkeypatch.setattr(orekit_utils, "start_orekit", _start)
    with pytest.raises(_StartAttempted):
        jb2008_density(positions, TIMES, **kwargs)
    assert not jpype.isJVMStarted()
