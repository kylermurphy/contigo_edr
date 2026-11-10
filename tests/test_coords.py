"""Tests for ``contigo.contigo_utils.coords`` (CD-01 .. CD-05, FX-01 .. FX-08).

Geodetic expectations come from the WGS-84 definition (a, 1/f) and its derived
semi-minor axis b = a(1 - f). Frame expectations come from Earth's rotation
(an Earth-fixed point on the equator moves at omega * a ~ 465.1 m/s inertially)
and from Orekit's ECI state stored in ``orekit_ref_small``.

added: 10/11/2026 Claude Code
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
import spiceypy as spice

import contigo.config as config
from contigo.contigo_utils import kernels
from contigo.contigo_utils.constants import WGS84_A, WGS84_F
from contigo.contigo_utils.coords import (
    ecef_to_geodetic,
    ecef_to_inertial,
    geodetic_to_ecef,
    inertial_to_ecef,
)
from contigo.contigo_utils.time_utils import spice_time

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


# ---------------------------------------------------------------------------
# FX: ECEF <-> inertial
# ---------------------------------------------------------------------------
def test_fx01_equatorial_surface_speed():
    """FX-01: a point fixed to Earth moves at ~omega * a = 465.1 m/s inertially."""
    out = ecef_to_inertial([[WGS84_A, 0, 0, 0, 0, 0]], [ET_2018])
    assert np.linalg.norm(out[0, 3:]) == pytest.approx(465.1, abs=0.1)


def test_fx02_round_trip():
    """FX-02: ECEF -> inertial -> ECEF returns the input to < 1e-6 m."""
    rng = np.random.default_rng(1)
    st = np.concatenate([rng.normal(0, 7e6, (50, 3)), rng.normal(0, 7e3, (50, 3))], 1)
    et = ET_2018 + np.linspace(0, 86400, 50)
    back = inertial_to_ecef(ecef_to_inertial(st, et), et)
    assert np.abs(back - st).max() < 1e-6


def test_fx03_position_only_matches_state():
    """FX-03: (N, 3) input equals the first three columns of the (N, 6) result."""
    st = np.array([[7e6, 1e5, 2e5, 10.0, 7000.0, 100.0]] * 3)
    et = ET_2018 + np.array([0.0, 60.0, 120.0])
    np.testing.assert_allclose(
        ecef_to_inertial(st[:, :3], et), ecef_to_inertial(st, et)[:, :3], atol=1e-9
    )


@pytest.mark.parametrize(
    "state, et",
    [
        (np.zeros((3, 4)), np.zeros(3)),  # wrong width
        (np.zeros((3, 6)), np.zeros(2)),  # length mismatch
        (np.zeros((3, 6)), np.array([0.0, np.nan, 1.0])),  # non-finite time
    ],
)
def test_fx04_bad_input(state, et):
    """FX-04: bad shapes or non-finite times raise ValueError."""
    with pytest.raises(ValueError):
        ecef_to_inertial(state, et)


def test_fx05_matches_orekit_eci(orekit_ref_small, esa_pod_small):
    """FX-05: CONTIGO (SPICE ITRF93) vs the stored Orekit ECI state.

    Measured: max 0.81 m / 0.9 mm/s over two orbits (ITRF93/J2000 in SPICE vs
    ITRF/EME2000 in Orekit); asserted at 5 m and 5 mm/s.
    """
    df = esa_pod_small
    et = spice_time(pd.DatetimeIndex(pd.to_datetime(df["DateTime"])), "GPS", "ET")
    st = df[["x", "y", "z", "vx", "vy", "vz"]].to_numpy() * 1e3  # file is km
    ref = orekit_ref_small[
        ["eci_x", "eci_y", "eci_z", "eci_vx", "eci_vy", "eci_vz"]
    ].to_numpy()
    out = ecef_to_inertial(st, et)
    assert np.linalg.norm(out[:, :3] - ref[:, :3], axis=1).max() < 5.0
    assert np.linalg.norm(out[:, 3:] - ref[:, 3:], axis=1).max() < 5e-3


def test_fx06_uncovered_epoch_warns_and_uses_iau_earth(offline):
    """FX-06: epochs after the PCK end warn once per call and use IAU_EARTH."""
    st = np.array([[7e6, 0, 0, 0, 7500.0, 0]])
    with pytest.warns(RuntimeWarning, match="IAU_EARTH"):
        out = ecef_to_inertial(st, [ET_2026_09])
    assert np.linalg.norm(out[0, :3]) == pytest.approx(7e6, rel=1e-9)


def test_fx07_uncovered_epoch_raises(offline):
    """FX-07: fallback='raise' raises RuntimeError naming the coverage."""
    with pytest.raises(RuntimeError, match="2026-05-30"):
        ecef_to_inertial(np.zeros((1, 6)), [ET_2026_09], fallback="raise")


def test_fx08_iau_earth_is_close_in_covered_range(offline):
    """FX-08: guard on the fallback quality, not a claim of accuracy.

    Measured against ITRF93 (2018 and 2026-05-29): 8 to 10 km, 9 m/s at LEO.
    Asserted at 20 km / 20 m/s so a wrong frame (e.g. a lost rotation, ~34 km
    for a 69 s error) fails.
    """
    kernels.furnsh_kernel(kernels.ensure_kernel(config.TPC_FILE, check_remote=False))
    st = np.array([[7e6, 0, 0, 0, 7500.0, 0]])
    good = ecef_to_inertial(st, [ET_2018])
    approx = np.einsum("ij,j->i", spice.sxform("IAU_EARTH", "J2000", ET_2018), st[0])
    assert np.linalg.norm(approx[:3] - good[0, :3]) < 20e3
    assert np.linalg.norm(approx[3:] - good[0, 3:]) < 20.0
