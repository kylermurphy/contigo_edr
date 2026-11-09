"""Third-body acceleration tests (PR T5, rows FO-01, FO-02, FO-03, FO-12, FO-14).

Physics
-------
The third-body (tidal) acceleration of a satellite at ``r`` due to a body at
``R`` (both geocentric) with mass parameter ``mu`` is

    a = mu * ( (R - r) / |R - r|^3  -  R / |R|^3 ),

the direct attraction on the satellite minus the attraction on the Earth
(Montenbruck & Gill, *Satellite Orbits*, 2000, eq. 3.37). To first order in
``r/d`` (``d = |R|``) its magnitude is ``mu r / d^3 * sqrt(1 + 3 cos^2 theta)``,
i.e. between ``mu r / d^3`` and ``2 mu r / d^3``.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from contigo.contigo_utils.constants import jpl_de440
from contigo.ephemeris.spice_ephem import SPICEEphem
from contigo.forces.tba_utils import tba_pairwise_numba
from contigo.forces.third_body_acc import ThirdBodyAcc, ThirdBodyEnv
from contigo.solar_system_ephem import SolarSystemEnvironment

BODIES = ["SUN", "MOON"]
GPS_MINUS_UTC_2018 = pd.Timedelta(seconds=18)

#: Row 0 of the SPICE third-body accelerations printed in the
#: ``testing_forces`` notebook (cells 15, 18), m/s^2, 9 significant digits.
SNAPSHOT_ROW0 = {
    "SUN": np.array([-9.00593868e-08, 2.80000065e-07, 9.67684671e-09]),
    "MOON": np.array([-1.81861563e-07, 7.29546172e-07, 5.87371096e-08]),
}


def _tba_numpy(r_sat, r_bodies, mu_bodies):
    """Pure NumPy reference for ``tba_pairwise_numba`` (same formula)."""
    rho = r_bodies - r_sat[None, :, :]
    rho3 = np.linalg.norm(rho, axis=2, keepdims=True) ** 3
    rb3 = np.linalg.norm(r_bodies, axis=2, keepdims=True) ** 3
    return mu_bodies[:, None, None] * (rho / rho3 - r_bodies / rb3)


@pytest.fixture
def spice_env() -> SolarSystemEnvironment:
    """Sun + Moon from the bundled de440s kernel, ITRF93, Earth-centred."""
    return SolarSystemEnvironment(
        BODIES,
        tolerance=0.01,
        provider=SPICEEphem(ephemeris="de440s", frame="ITRF93", observer="EARTH"),
    )


def _assert_epochs_align(sc, orekit_ref_small):
    """The Orekit rows are labelled in GPS time; ``sc_utc`` is UTC (tz-aware)."""
    utc = pd.DatetimeIndex(sc.sc_utc).tz_convert(None)
    gps = pd.DatetimeIndex(orekit_ref_small["DateTime"])
    assert len(utc) == len(gps)
    assert (utc + GPS_MINUS_UTC_2018 == gps).all()


# ---------------------------------------------------------------------------
# FO-01 kernel
# ---------------------------------------------------------------------------
def test_tba_pairwise_numba_matches_numpy_reference():
    """FO-01: numba kernel equals a NumPy implementation on random inputs.

    The result is a difference of two terms of size ``mu / |R - r|^2`` and
    ``mu / |R|^2``. Where a component nearly cancels, rounding of those terms
    (the kernel uses ``fastmath``, so operation order differs) dominates, so
    the check is ``|got - want| <= 1e-12 |want| + 1e-14 * term size``: the
    catalogue's rtol 1e-12 plus an absolute floor of ~50 ulp of the terms.
    Pure rtol 1e-12 failed on 14 of 2313 near-cancelling components (max
    relative 1.6e-11, absolute 2.7e-18 m/s^2 against terms of ~3e-5).
    """
    rng = np.random.default_rng(5)
    n, b = 257, 3
    r_sat = rng.normal(size=(n, 3)) * 7e6
    r_bodies = rng.normal(size=(b, n, 3)) * 4e8
    mu = np.array([jpl_de440["MOON"], jpl_de440["EARTH"], 1e15])

    got = tba_pairwise_numba(r_sat, r_bodies, mu)
    want = _tba_numpy(r_sat, r_bodies, mu)

    assert got.shape == (b, n, 3)
    rho = np.linalg.norm(r_bodies - r_sat[None], axis=2, keepdims=True)
    rb = np.linalg.norm(r_bodies, axis=2, keepdims=True)
    term = mu[:, None, None] * (1 / rho**2 + 1 / rb**2)
    err = np.abs(got - want)
    assert (err <= 1e-12 * np.abs(want) + 1e-14 * term).all()
    # the bulk of components agree to the catalogue's rtol 1e-12
    assert np.mean(err <= 1e-12 * np.abs(want)) > 0.98


@pytest.mark.parametrize(
    "direction", [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, -2.0, 0.5]]
)
def test_tba_pairwise_numba_collinear_analytic(direction):
    """FO-01: satellite on the Earth-body line, same side as the body.

    With ``r = r u`` and ``R = D u`` the formula reduces exactly to
    ``a = mu (1/(D-r)^2 - 1/D^2) u``, pointing toward the body.
    """
    u = np.asarray(direction) / np.linalg.norm(direction)
    mu = jpl_de440["MOON"]
    D, r = 3.844e8, 7.0e6
    r_sat = (r * u)[None, :]
    r_body = (D * u)[None, None, :]

    a = tba_pairwise_numba(r_sat, r_body, np.array([mu]))[0, 0]

    expected = mu * (1.0 / (D - r) ** 2 - 1.0 / D**2)
    np.testing.assert_allclose(np.linalg.norm(a), expected, rtol=1e-10)
    np.testing.assert_allclose(a / np.linalg.norm(a), u, atol=1e-12)  # toward body


def test_tba_pairwise_numba_satellite_at_origin_is_zero():
    """FO-01: at the geocentre direct and indirect terms cancel.

    Mathematically the result is exactly zero. With ``fastmath`` the two
    algebraically identical terms (``mu / D^2``: 6e-3 m/s^2 for the Sun,
    3e-5 for the Moon) are rounded differently, leaving ~3e-19 m/s^2, so
    zero is asserted to 1e-14 of the term size.
    """
    r_bodies = np.array([[[1.4e11, -2.0e10, 5.0e10]], [[3.0e8, 1.0e8, -1.2e8]]])
    mu = np.array([jpl_de440["SUN"], jpl_de440["MOON"]])
    a = tba_pairwise_numba(np.zeros((1, 3)), r_bodies, mu)
    term = mu / np.linalg.norm(r_bodies[:, 0], axis=1) ** 2
    assert (np.abs(a[:, 0]) <= 1e-14 * term[:, None]).all()


# ---------------------------------------------------------------------------
# FO-02 ThirdBodyEnv with a fake provider
# ---------------------------------------------------------------------------
def test_third_body_env_with_fake_provider(constellation_fixture, fake_provider):
    """FO-02: one ``(B, N, 3)`` array per constellation member.

    Also checks the result equals the kernel applied to the provider's
    positions, so the per-member time slicing is right.
    """
    env = SolarSystemEnvironment(BODIES, tolerance=0.01, provider=fake_provider)
    acc = ThirdBodyEnv().acceleration(constellation_fixture, env)

    assert set(acc) == set(constellation_fixture.spacecraft)
    for sc_id, sc in constellation_fixture.spacecraft.items():
        assert acc[sc_id].shape == (2, sc.N, 3)
        assert np.isfinite(acc[sc_id]).all()
        r_bodies = fake_provider(BODIES, ephem_time=sc.sspice_et)
        want = tba_pairwise_numba(sc.state_ecef[:, :3], r_bodies, env.GM)
        np.testing.assert_allclose(acc[sc_id], want, rtol=1e-14)


# ---------------------------------------------------------------------------
# FO-03 real SPICE magnitudes
# ---------------------------------------------------------------------------
def test_third_body_env_spice_magnitudes(esa_constellation, spice_env):
    """FO-03: Sun and Moon tidal accelerations at LEO are 1e-7 to 2e-6 m/s^2.

    Derivation: ``|a| = mu r / d^3 sqrt(1 + 3 cos^2 theta)`` to first order.
    With r ~ 6.9e6 m:
    Sun (mu 1.327e20, d ~ 1.47e11 m in January): 2.9e-7 to 5.7e-7 m/s^2;
    Moon (mu 4.903e12, d 3.6e8 to 4.06e8 m): 5.0e-7 to 1.45e-6 m/s^2.
    Each sample is also checked against its own first-order bounds
    ``[1, 2] * mu r / d^3`` with 3 % slack for the O(r/d) terms (r/d ~ 2 %
    for the Moon).
    """
    sc = esa_constellation.spacecraft["ESA"]
    acc = ThirdBodyEnv().acceleration(esa_constellation, spice_env)["ESA"]
    *_, r_bodies = spice_env.get_ephem(sc.sspice_et, sc.sspice_gps, sc.sc_utc)

    mag = np.linalg.norm(acc, axis=2)
    assert acc.shape == (2, sc.N, 3)
    assert (mag > 1e-7).all() and (mag < 2e-6).all()

    r = np.linalg.norm(sc.state_ecef[:, :3], axis=1)
    d = np.linalg.norm(r_bodies, axis=2)
    scale = spice_env.GM[:, None] * r[None, :] / d**3
    assert (mag > 0.97 * scale).all()
    assert (mag < 2.0 * 1.03 * scale).all()


# ---------------------------------------------------------------------------
# FO-14 SPICE vs stored Orekit results
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ibody, col, rel_tol",
    [
        # measured max |a_spice - a_orekit| / |a_orekit| = 8.0e-8
        (0, "sn", 2e-7),
        # measured 3.7e-7, nearly constant: implies a lunar GM ~3.4e-7 smaller
        # in the Orekit run (recomputing with Orekit's Moon positions still
        # gives 3.4e-7, so it is not the ephemeris)
        (1, "mn", 1e-6),
    ],
    ids=["SUN", "MOON"],
)
def test_third_body_env_matches_orekit(
    esa_constellation, orekit_ref_small, spice_env, ibody, col, rel_tol
):
    """FO-14: SPICE ``ThirdBodyEnv`` vs the stored Orekit accelerations.

    The catalogue's ``allclose`` default (atol 1e-8) is vacuous against
    accelerations of ~3e-7 m/s^2, and elementwise rtol fails on near-zero
    components. Instead the vector difference is compared with the vector
    norm, per sample. Expected differences: SPICE ITRF93 and Orekit's ITRF
    differ by a ~3e-8 rad rotation (Sun positions ~4.5 km apart, distances
    equal to mm), which moves the tidal vector by a few times 3e-8 relative.
    """
    sc = esa_constellation.spacecraft["ESA"]
    _assert_epochs_align(sc, orekit_ref_small)

    acc = ThirdBodyEnv().acceleration(esa_constellation, spice_env)["ESA"][ibody]
    ref = orekit_ref_small[[f"ecef_{col}_a{c}" for c in "xyz"]].to_numpy()

    rel = np.linalg.norm(acc - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert rel.max() < rel_tol, f"max relative difference {rel.max():.3g}"


def test_third_body_env_row0_snapshot(esa_constellation, spice_env):
    """FO-14: row 0 equals the notebook snapshot (9 significant digits).

    The snapshot was printed with 9 significant digits, so each component
    carries a rounding error of at most 5e-9 relative; rtol 1e-8 is used.
    """
    acc = ThirdBodyEnv().acceleration(esa_constellation, spice_env)["ESA"]
    for i, body in enumerate(BODIES):
        np.testing.assert_allclose(acc[i, 0], SNAPSHOT_ROW0[body], rtol=1e-8, atol=0)


# ---------------------------------------------------------------------------
# FO-12 legacy ThirdBodyAcc
# ---------------------------------------------------------------------------
def _legacy(sc, stime=None):
    if stime is None:
        # GPS labels, tz-naive (DatetimeIndex: see the Series xfail below)
        stime = pd.DatetimeIndex(sc.sc_utc).tz_convert(None) + GPS_MINUS_UTC_2018
    return ThirdBodyAcc(
        spos=sc.state_ecef[:, :3], stime=stime, body=BODIES, scale="GPS"
    )


def test_third_body_acc_legacy_equals_env(esa_constellation, spice_env):
    """FO-12: legacy ``ThirdBodyAcc`` equals ``ThirdBodyEnv`` bit for bit.

    Both evaluate ``spkpos`` at the same ET (the legacy class converts GPS to
    ET with ``unitim``; ``Spacecraft`` does the same) and call the same
    kernel, so no tolerance is needed (as in the notebook). The environment's
    0.01 s binning does not alter positions here because the 10 s samples
    fall in distinct bins.
    """
    sc = esa_constellation.spacecraft["ESA"]
    env_acc = ThirdBodyEnv().acceleration(esa_constellation, spice_env)["ESA"]

    leg = _legacy(sc)
    leg.calc_tba()

    np.testing.assert_array_equal(leg.get_tba(), env_acc)
    assert leg.get_body_pos().shape == (2, sc.N, 3)


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="R9: ThirdBodyAcc not yet deprecated",
)
def test_third_body_acc_warns_deprecation(esa_constellation):
    """FO-12: after R9, using ``ThirdBodyAcc`` emits ``DeprecationWarning``."""
    sc = esa_constellation.spacecraft["ESA"]
    with pytest.warns(DeprecationWarning):
        leg = _legacy(sc)
        leg.calc_tba()


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason=(
        "NEW: ThirdBodyAcc.calc_tba reads stime.tzinfo, which a pandas Series "
        "lacks; fails for a Series stime, including the default"
    ),
)
def test_third_body_acc_accepts_series_stime(esa_constellation, spice_env):
    """FO-12: ``stime`` given as a ``pd.Series`` (the documented default type).

    ``pd.to_datetime(Series)`` returns a Series, which has ``.dt.tz`` but no
    ``.tzinfo``.
    """
    sc = esa_constellation.spacecraft["ESA"]
    stime = pd.Series(pd.DatetimeIndex(sc.sc_utc).tz_convert(None) + GPS_MINUS_UTC_2018)
    leg = _legacy(sc, stime=stime)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        leg.calc_tba()
    env_acc = ThirdBodyEnv().acceleration(esa_constellation, spice_env)["ESA"]
    np.testing.assert_array_equal(leg.get_tba(), env_acc)
