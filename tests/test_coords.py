"""Tests for ``contigo.contigo_utils.coords``: geodetic conversion (CD-01 .. CD-05).

Expectations come from the WGS-84 definition (a, 1/f) and its derived
semi-minor axis b = a(1 - f).

added: 10/11/2026 Claude Code
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import contigo.config as config
from contigo.contigo_utils.constants import WGS84_A, WGS84_F
from contigo.contigo_utils.coords import (
    ecef_to_geodetic,
    geodetic_to_ecef,
)

WGS84_B = WGS84_A * (1.0 - WGS84_F)
ET_2018 = 5.68123251e8  # inside the bundled PCK coverage
ET_2026_09 = 8.4e8  # 2026-09, after the bundled PCK ends (2026-05-30)


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No network, empty cache, and a fresh once-per-session refresh flag."""
    monkeypatch.setenv("CONTIGO_OFFLINE", "1")
    monkeypatch.setenv("CONTIGO_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(config.state, "pck_refreshed", False)


# ---------------------------------------------------------------------------
# CD: geodetic conversion (pure NumPy)
# ---------------------------------------------------------------------------
def test_cd01_equator_pole_and_altitude():
    """CD-01: points on the ellipsoid have alt 0; +400 km radially adds 400 km."""
    r = np.array([[WGS84_A, 0, 0], [0, 0, WGS84_B], [0, 0, -WGS84_B]])
    lat, lon, alt = ecef_to_geodetic(r).T
    assert alt == pytest.approx(0.0, abs=1e-6)
    assert lat == pytest.approx([0.0, np.pi / 2, -np.pi / 2])
    leo = ecef_to_geodetic([[WGS84_A + 400e3, 0.0, 0.0]])[0]
    assert leo[2] == pytest.approx(400e3, abs=1e-3)


@pytest.mark.parametrize("alt_range", [(-100e3, 2000e3), (0.0, 40000e3)])
def test_cd02_round_trip(alt_range):
    """CD-02: geodetic -> ECEF -> geodetic recovers altitude to < 1e-6 m."""
    rng = np.random.default_rng(0)
    n = 100_000
    llh = np.stack(
        [
            np.arcsin(rng.uniform(-1, 1, n)),  # uniform on the sphere
            rng.uniform(-np.pi, np.pi, n),
            rng.uniform(*alt_range, n),
        ],
        axis=-1,
    )
    back = ecef_to_geodetic(geodetic_to_ecef(llh))
    assert np.abs(back[:, 2] - llh[:, 2]).max() < 1e-6
    assert np.abs(back[:, 0] - llh[:, 0]).max() < 1e-12


def test_cd03_bad_shape():
    """CD-03: last axis must be 3."""
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        ecef_to_geodetic(np.zeros((10, 2)))


def test_cd04_leading_axes_preserved():
    """CD-04: (..., 3) in -> (..., 3) out."""
    assert ecef_to_geodetic(np.full((4, 5, 3), 7e6)).shape == (4, 5, 3)


def test_cd05_million_points_fast():
    """CD-05: 1e6 points in < 2 s (measured 0.2 s; acceptance of plotting-plan task 0.2)."""
    r = geodetic_to_ecef(np.tile([0.3, 1.0, 500e3], (1_000_000, 1)))
    t0 = time.perf_counter()
    ecef_to_geodetic(r)
    assert time.perf_counter() - t0 < 2.0
