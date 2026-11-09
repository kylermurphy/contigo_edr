"""Tests for ``contigo.edr_efd.EDRDensity`` (PR T6, catalogue §4.6 ED-01..ED-09).

Method: doi:10.1029/2024EA003898, equations A18-A20. ``compute_edr`` returns,
per spacecraft,

    EDR(t) = J(t) - J(0) - sum_models int_0^t a_m . v dt'
    J      = |v|^2 / 2 - U - w^2 (x^2 + y^2) / 2

(ECEF, SI; ``U`` positive, as ``EarthPotential`` returns it; the integral is
``scipy.integrate.cumulative_simpson`` over GPS seconds). ``compute_denom``
returns ``int_0^t b |v|^3 dt'`` with ``b = Cd A / m``, and ``compute_den``
returns ``efd = -2 dEDR / d(denom)`` over windows.

The physics tests (ED-01..ED-05) use exact circular Keplerian orbits (see
``tests/_support.make_circular_orbit``), for which ``J`` is conserved under a
point-mass potential, so every expected value is analytic.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _support import (
    ESA_PROPS,
    GM_EARTH,
    OMEGA,
    R_EARTH,
    FakePotential,
    FakeProvider,
    make_circular_orbit,
)

from contigo.constellation import Constellation
from contigo.edr_efd import EDRDensity
from contigo.ephemeris.spice_ephem import SPICEEphem
from contigo.forces.grav_pot import EarthPotential
from contigo.forces.third_body_acc import ThirdBodyEnv
from contigo.solar_system_ephem import SolarSystemEnvironment
from contigo.spacecraft import Spacecraft

DATA = Path(__file__).resolve().parent / "data"
REPO = Path(__file__).resolve().parents[1]

N = 600  # samples of the synthetic orbits
DT = 10.0  # s
B_ESA = ESA_PROPS["cd"] * ESA_PROPS["drag_area"] / ESA_PROPS["sc_mass"]  # m^2/kg


# ---------------------------------------------------------------------------
# Helpers and local force doubles
# ---------------------------------------------------------------------------
def _constellation(inc_deg: float, n: int = N, sc_id: str = "X") -> Constellation:
    state, time = make_circular_orbit(inc_deg=inc_deg, n=n, dt=DT)
    sc = Spacecraft(
        state=state,
        time=time,
        sc_id_input=np.full(n, sc_id),
        tscale_input="GPS",
        **ESA_PROPS,
    )
    return Constellation(spacecraft=sc)


def _fake_env(provider: FakeProvider | None = None) -> SolarSystemEnvironment:
    return SolarSystemEnvironment(
        ["SUN", "MOON"], tolerance=0.01, provider=provider or FakeProvider()
    )


def _elapsed(con: Constellation, sc_id: str = "X") -> np.ndarray:
    gps = con[sc_id].sspice_gps
    return gps - gps[0]


def _speed(con: Constellation, sc_id: str = "X") -> np.ndarray:
    return np.linalg.norm(con[sc_id].state_ecef[:, 3:], axis=1)


class LinearDrag:
    """``a = -k v`` (N, 3), with a configurable name."""

    def __init__(self, k: float, name: str) -> None:
        self.k = k
        self.name = name

    def acceleration(self, constellation, solarsys_env) -> dict:
        return {
            i: -self.k * sc.state_ecef[:, 3:]
            for i, sc in constellation.spacecraft.items()
        }

    def potential(self, constellation):
        raise NotImplementedError


class ZPush:
    """``a = c z_hat`` (N, 3): ``int a . v dt = c (z(t) - z(0))``, time-varying.

    (A radial push would be useless here: ``r_hat . v_ecef = 0`` on a circular
    orbit, so it does no work.)
    """

    def __init__(self, c: float, name: str) -> None:
        self.c = c
        self.name = name

    def acceleration(self, constellation, solarsys_env) -> dict:
        out = {}
        for i, sc in constellation.spacecraft.items():
            a = np.zeros((sc.N, 3))
            a[:, 2] = self.c
            out[i] = a
        return out

    def potential(self, constellation):
        raise NotImplementedError


class StackedForce:
    """Stacks several (N, 3) forces into one (B, N, 3) result, like ThirdBodyEnv."""

    name = "Stacked"

    def __init__(self, *forces) -> None:
        self.forces = forces

    def acceleration(self, constellation, solarsys_env) -> dict:
        parts = [f.acceleration(constellation, solarsys_env) for f in self.forces]
        return {i: np.stack([p[i] for p in parts]) for i in parts[0]}

    def potential(self, constellation):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Physics: compute_edr
# ---------------------------------------------------------------------------
def test_edr_zero_for_keplerian_orbit_without_forces():
    """ED-01: Jacobi integral conservation gives EDR = 0.

    For a Keplerian orbit under the point-mass potential ``U = GM/r``, the
    Jacobi integral of the rotating (ECEF) frame,
    ``J = |v|^2/2 - GM/r - w^2 (x^2 + y^2)/2``, is constant. With no force
    models ``EDR(t) = J(t) - J(0) = 0``. Tolerance 1e-6 m^2/s^2: ``J`` is a
    difference of terms of size ~3e7 m^2/s^2, so round-off is ~1e-8; the orbit
    is inclined (51.6 deg) so ``|v_ecef|`` and ``x^2 + y^2`` both vary and the
    cancellation between the three terms is actually exercised.
    """
    con = _constellation(inc_deg=51.6)
    edr = EDRDensity(con, _fake_env(), [], FakePotential()).compute_edr()
    assert set(edr) == {"X"}
    assert edr["X"].shape == (N,)
    assert edr["X"][0] == 0.0
    # the terms really do vary along this orbit (else the test is vacuous)
    rho2 = con["X"].state_ecef[:, 0] ** 2 + con["X"].state_ecef[:, 1] ** 2
    assert np.ptp(OMEGA**2 * rho2 / 2) > 1e4
    assert np.abs(edr["X"]).max() < 1e-6


def test_edr_linear_drag_equatorial_orbit():
    """ED-02: ``a = -k v`` on an equatorial circular orbit gives EDR = k |v|^2 t.

    On an equatorial circular orbit ``|v_ecef| = (n_m - w) a`` is constant and
    ``J`` is conserved (ED-01). The code computes
    ``EDR = J - J(0) - int a . v dt``; with ``a = -k v`` the integrand is the
    constant ``-k |v|^2``, so ``int a . v dt = -k |v|^2 t`` and
    ``EDR(t) = +k |v|^2 t`` (a modelled dissipative force that is absent from
    the orbit appears as a positive EDR).

    Simpson's rule is exact for a constant integrand, so the only error is
    the round-off of ``J`` (~1e-8 m^2/s^2, see ED-01). With k = 1e-6 1/s the
    signal is k |v|^2 dt ~ 0.5 m^2/s^2 after one step and ~3e5 at the end, so
    ``atol=1e-6`` (ED-01 bound) plus ``rtol=1e-10`` leaves margin of ~100 over
    the round-off and is still 1e5 times smaller than the first-step signal.
    """
    k = 1e-6
    con = _constellation(inc_deg=0.0)
    v = _speed(con)
    a = R_EARTH + 400e3
    np.testing.assert_allclose(v, (np.sqrt(GM_EARTH / a**3) - OMEGA) * a, rtol=1e-12)
    edr = EDRDensity(
        con, _fake_env(), [LinearDrag(k, "Drag")], FakePotential()
    ).compute_edr()["X"]
    expected = k * v**2 * _elapsed(con)
    assert edr[-1] > 0  # sign convention
    np.testing.assert_allclose(edr, expected, rtol=1e-10, atol=1e-6)


def test_edr_stacked_force_equals_separate_forces():
    """ED-03: one (B, N, 3) force equals B separate (N, 3) forces.

    ``compute_edr`` integrates each body of a 3-D result separately and adds
    the integrals, which is exactly what it does for separate 2-D forces, so
    the EDR must be identical (rtol 1e-12, i.e. round-off only). The two
    components are a linear drag and a constant z push (whose ``a . v`` varies in
    time on the inclined orbit), so a dropped or mis-indexed body changes the
    result by far more than the tolerance.
    """
    con = _constellation(inc_deg=51.6)
    f1, f2 = LinearDrag(1e-6, "Drag"), ZPush(1e-3, "ZPush")
    sep = EDRDensity(con, _fake_env(), [f1, f2], FakePotential()).compute_edr()["X"]
    stk = EDRDensity(
        con, _fake_env(), [StackedForce(f1, f2)], FakePotential()
    ).compute_edr()["X"]
    only1 = EDRDensity(con, _fake_env(), [f1], FakePotential()).compute_edr()["X"]
    # the z push contributes -int a.v dt = -c (z - z0) (~5e3 m^2/s^2); Simpson
    # error on this smooth sinusoid (period ~5550 s, dt 10 s) is ~1e-10 relative
    z = con["X"].state_ecef[:, 2]
    np.testing.assert_allclose(only1 - sep, 1e-3 * (z - z[0]), rtol=1e-7, atol=1e-6)
    assert np.abs(sep - only1).max() > 1e3  # the second component matters
    np.testing.assert_allclose(stk, sep, rtol=1e-12, atol=1e-9)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="NEW: compute_edr keys accelerations by model.name, so two force "
    "models with the same name silently overwrite each other (edr_efd.py "
    "acc_con[model.name] = ...) and only the last one is integrated",
)
def test_edr_two_forces_with_same_name_both_count():
    """ED-03: two (N, 3) forces sharing a ``name`` are both integrated.

    Two ``a = -k v`` forces with the same name must act as one force with
    ``2k``: ``EDR = 2 k |v|^2 t`` on the equatorial orbit (see ED-02).
    """
    k = 1e-6
    con = _constellation(inc_deg=0.0)
    forces = [LinearDrag(k, "Drag"), LinearDrag(k, "Drag")]
    edr = EDRDensity(con, _fake_env(), forces, FakePotential()).compute_edr()["X"]
    expected = 2 * k * _speed(con) ** 2 * _elapsed(con)
    np.testing.assert_allclose(edr, expected, rtol=1e-10, atol=1e-6)


# ---------------------------------------------------------------------------
# Physics: compute_denom and compute_den
# ---------------------------------------------------------------------------
def test_denom_constant_b_and_speed():
    """ED-04: ``denom(t) = int_0^t b |v|^3 dt' = b |v|^3 t``.

    With constant ``b = Cd A / m`` (3.5 * 1.1 / 430) and constant ``|v|``
    (equatorial circular orbit) the integrand is constant and Simpson's rule
    is exact; the only error is round-off (~1e-15 relative), hence rtol 1e-12.
    """
    con = _constellation(inc_deg=0.0)
    denom = EDRDensity(con, _fake_env(), [], FakePotential()).compute_denom()
    assert set(denom) == {"X"}
    v = _speed(con)
    expected = B_ESA * v**3 * _elapsed(con)
    assert denom["X"][0] == 0.0
    np.testing.assert_allclose(denom["X"], expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "n, n_windows",
    [
        (600, 6),  # T = 5990 s: floor(5990 / 900) = 6
        (541, 6),  # T = 5400 s: last window ends exactly on the last sample
        (540, 5),  # T = 5390 s: floor(5390 / 900) = 5
    ],
)
def test_den_windows_midpoints_and_value(n, n_windows):
    """ED-05: windows, midpoints and ``efd = -2k / (b |v|)``.

    Window loop: ``s = t0 + j W``, ``e = s + W`` while ``e <= t_last``, so the
    count is ``floor(T / W)`` with ``T = t_last - t0``. With W = 900 s and
    dt = 10 s, ``i0 = searchsorted(s, 'left') = 90 j`` and
    ``i1 = searchsorted(e, 'left') - 1 = 90 j + 89`` (the sample at ``e``
    itself is excluded, so each window spans W - dt = 890 s). The DateTime is
    the midpoint ``t[i0] + (t[i1] - t[i0]) / 2 = t0 + 900 j + 445 s``.

    With ED-02/ED-04 inputs, dEDR = k |v|^2 dt and d(denom) = b |v|^3 dt over
    the same interval, so ``efd = -2 k / (b |v|)`` in every window,
    independent of the window length. ``smth_edr=1`` disables the rolling
    mean (a centred mean of a linear function is unchanged in the interior,
    but not at the end points, which would bias the first and last windows).
    rtol 1e-9: both deltas carry the ED-02/ED-04 round-off (<1e-10 relative).
    """
    k = 1e-6
    window = pd.Timedelta(seconds=900)
    con = _constellation(inc_deg=0.0, n=n)
    v = _speed(con)[0]
    calc = EDRDensity(con, _fake_env(), [LinearDrag(k, "Drag")], FakePotential())
    efd = calc.compute_den(window=window, smth_edr=1)
    assert set(efd) == {"X"}
    df = efd["X"]
    assert list(df.columns) == ["DateTime", "efd"]

    t = pd.DatetimeIndex(con["X"].sc_utc)
    span = (t[-1] - t[0]).total_seconds()
    assert len(df) == n_windows == int(span // window.total_seconds())

    j = np.arange(n_windows)
    i0, i1 = 90 * j, 90 * j + 89
    mid = t[i0] + (t[i1] - t[i0]) / 2
    assert (pd.DatetimeIndex(df["DateTime"]) == mid).all()
    assert (mid == t[0] + pd.to_timedelta(900 * j + 445, unit="s")).all()

    expected = -2 * k / (B_ESA * v)
    np.testing.assert_allclose(df["efd"].to_numpy(), expected, rtol=1e-9)
    # the intermediate results are stored on the instance
    assert set(calc.edr) == set(calc.denom) == {"X"}


def test_den_mismatched_keys_raise():
    """ED-05: EDR and denominator keyed by different IDs raise ValueError."""
    con = _constellation(inc_deg=0.0)
    calc = EDRDensity(con, _fake_env(), [], FakePotential())
    good = calc.compute_denom()["X"]
    calc.compute_edr = lambda: {"X": np.zeros_like(good)}
    calc.compute_denom = lambda: {"Y": good}
    with pytest.raises(ValueError, match="keys do not match"):
        calc.compute_den(window=pd.Timedelta(seconds=900))


def test_den_mismatched_shapes_raise():
    """ED-05: EDR and denominator of different lengths raise ValueError."""
    con = _constellation(inc_deg=0.0)
    calc = EDRDensity(con, _fake_env(), [], FakePotential())
    good = calc.compute_denom()["X"]
    calc.compute_edr = lambda: {"X": np.zeros(good.size - 1)}
    calc.compute_denom = lambda: {"X": good}
    with pytest.raises(ValueError, match="shapes do not match"):
        calc.compute_den(window=pd.Timedelta(seconds=900))


# ---------------------------------------------------------------------------
# Constructor and ephemeris cache
# ---------------------------------------------------------------------------
def test_constructor_warms_cache_with_one_provider_call():
    """ED-06: the constructor loads all epochs in exactly one provider call.

    ``EDRDensity.__init__`` calls ``solarsys_env._load_times`` once with the
    constellation's union time axis. A later ``compute_edr`` with
    ``ThirdBodyEnv`` on the same epochs must be served from the cache (no
    further provider call). Two spacecraft on the same epochs check that the
    union axis is de-duplicated (N cached bins, not 2N).
    """
    s1 = _constellation(inc_deg=51.6, sc_id="A")
    s2 = _constellation(inc_deg=87.0, sc_id="B")
    con = s1 + s2
    prov = FakeProvider()
    env = _fake_env(prov)
    assert prov.calls == 0
    calc = EDRDensity(con, env, [ThirdBodyEnv()], FakePotential())
    assert prov.calls == 1
    assert len(prov.requested[0]) == N
    assert len(env._cache) == N
    edr = calc.compute_edr()
    assert prov.calls == 1
    assert set(edr) == {"A", "B"}
    assert calc.accelerations[ThirdBodyEnv.name]["A"].shape == (2, N, 3)


# ---------------------------------------------------------------------------
# Real data: esa_pod_small vs Orekit
# ---------------------------------------------------------------------------
def test_denom_matches_orekit(esa_constellation, orekit_ref_small):
    """ED-09: ``compute_denom`` agrees with Orekit's ``denom`` (SI units).

    Units by inspection: after one 10 s step Orekit's value is 3.937274e10,
    and ``b |v|^3 dt = (3.5 * 1.1 / 430) * 7604^3 * 10 = 3.94e10`` with |v| in
    m/s, so the column is SI (m^2/kg * m^3/s^3 * s), and no 1000^5 factor is
    needed. The measured max relative difference is ~9.2e-7 (different
    integration/velocity handling in the Orekit pipeline), so rtol 1e-5 with
    atol 0 (tighter than ``np.allclose`` defaults) holds with margin ~10.
    """
    sc = esa_constellation["ESA"]
    # epochs align: POD is GPS time, sc_utc is UTC (GPS - 18 s in 2018)
    utc = pd.DatetimeIndex(sc.sc_utc).tz_convert(None) + pd.Timedelta(seconds=18)
    assert (utc == pd.DatetimeIndex(orekit_ref_small["DateTime"])).all()

    denom = EDRDensity(
        esa_constellation, _fake_env(), [], FakePotential()
    ).compute_denom()["ESA"]
    ore = orekit_ref_small["denom"].to_numpy()
    assert ore[0] == 0.0 and denom[0] == 0.0
    np.testing.assert_allclose(ore[1], B_ESA * 7604.0**3 * 10.0, rtol=2e-3)
    np.testing.assert_allclose(denom, ore, rtol=1e-5, atol=0)


def test_edr_regression_snapshot(esa_constellation, orekit_ref_small):
    """ED-07: EDR of ``esa_pod_small`` matches ``tests/data/edr_snapshot.npz``.

    Configuration: ThirdBodyEnv (Sun, Moon; SPICE de440s, tolerance 0.01 s)
    and EarthPotential(EIGEN-2, lmax 50); regenerate with
    ``tests/data/make_edr_snapshot.py``. This is a regression test, not a
    validation test. rtol 1e-9 rather than 1e-10: the potential uses numba
    with fastmath/parallel reductions, whose summation order can change across
    the CI platforms; on a ~6e7 m^2/s^2 potential a few ulp of reordering per
    term gives ~1e-8 m^2/s^2 absolute, which ``atol=1e-7`` covers relative to
    EDR values of order 1 m^2/s^2.

    For information only (no assertion): Orekit's ``edr`` includes SRP and
    its own potential; the max difference from this EDR is printed.
    """
    snap = np.load(DATA / "edr_snapshot.npz")
    expected = snap["edr"]
    assert expected.shape == (1104,) and expected.dtype == np.float64

    env = SolarSystemEnvironment(["SUN", "MOON"], 0.01, SPICEEphem())
    edr = EDRDensity(
        esa_constellation, env, [ThirdBodyEnv()], EarthPotential("EIGEN-2.gfc", 50)
    ).compute_edr()["ESA"]
    np.testing.assert_allclose(edr, expected, rtol=1e-9, atol=1e-7)

    ore = orekit_ref_small["edr"].to_numpy()
    diff = np.abs(edr - (ore - ore[0])).max()
    print(f"ED-07 info: max |EDR - Orekit EDR (with SRP)| = {diff:.3e} m^2/s^2")


@pytest.mark.orekit
@pytest.mark.slow
def test_edr_full_esa_pod_matches_orekit():
    """ED-08: full ``ESA_pod`` EDR (Sun, Moon, Orekit SRP, lmax 100) vs Orekit.

    The CONTIGO_Colab validation with ``SRPOrekitCB`` in place of GMAT SRP.
    Tolerance ``atol=1e-7, rtol=0.1`` is deliberately loose (decision of
    2026-11-09; to be tightened later). ``contigo.forces.srp_orekit`` is
    imported here because importing it starts the JVM.
    """
    from contigo.forces.srp_orekit import SRPOrekitCB

    pod = REPO / "data" / "ESA_pod.hdf"
    ore_file = REPO / "data" / "ore_d.hdf"
    con = Constellation(
        state_file=str(pod),
        time_col="DateTime",
        sc_id_input="ESA",
        tscale_input="GPS",
        unit_input="km",
        **ESA_PROPS,
    )
    assert con["ESA"].N == 138240
    env = SolarSystemEnvironment(["SUN", "MOON"], 0.01, SPICEEphem())
    edr = EDRDensity(
        con, env, [ThirdBodyEnv(), SRPOrekitCB()], EarthPotential(lmax=100)
    ).compute_edr()["ESA"]

    ore = pd.read_hdf(ore_file)["edr"].to_numpy()
    assert ore.shape == edr.shape
    assert np.allclose(edr, ore - ore[0], atol=1e-7, rtol=0.1)
