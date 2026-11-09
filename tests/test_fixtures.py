"""Checks on the shared fixtures themselves (PR T0).

The physics tests rely on ``circular_orbit`` being exact, so it is verified
here independently of any contigo computation.

added: 09/11/2026 Claude Code
"""

import numpy as np
import pandas as pd
import pytest
from _support import GM_EARTH, OMEGA, R_EARTH, FakeProvider, make_circular_orbit

from contigo.contigo_utils import utils


def test_circular_orbit_radius_and_speed():
    alt = 400e3
    state, time = make_circular_orbit(alt=alt, inc_deg=51.6, n=600)
    a = R_EARTH + alt
    r = np.linalg.norm(state[:, :3], axis=1)
    np.testing.assert_allclose(r, a, rtol=1e-14)
    # inertial speed sqrt(GM/a): add back w x r to the Earth-fixed velocity
    v_in = state[:, 3:] + np.cross([0.0, 0.0, OMEGA], state[:, :3])
    np.testing.assert_allclose(
        np.linalg.norm(v_in, axis=1), np.sqrt(GM_EARTH / a), rtol=1e-13
    )
    assert len(time) == 600
    assert (time.to_series().diff().dropna() == pd.Timedelta(seconds=10)).all()


def test_circular_orbit_velocity_is_derivative_of_position():
    """ECEF velocity matches a central difference of ECEF position."""
    dt = 1.0
    state, _ = make_circular_orbit(n=50, dt=dt, inc_deg=70.0)
    dr = (state[2:, :3] - state[:-2, :3]) / (2 * dt)
    v = state[1:-1, 3:]
    # central-difference error ~ |a| dt^2 / 6 ~ 1.5 mm/s
    np.testing.assert_allclose(dr, v, atol=5e-3)


def test_circular_orbit_conserves_jacobi_integral():
    """v^2/2 - GM/r - w^2 (x^2+y^2)/2 is constant in the rotating frame.

    This is the identity the EDR physics tests (ED-01) depend on.
    """
    state, _ = make_circular_orbit(inc_deg=51.6, n=600)
    r = np.linalg.norm(state[:, :3], axis=1)
    v2 = (state[:, 3:] ** 2).sum(axis=1)
    rho2 = state[:, 0] ** 2 + state[:, 1] ** 2
    jac = v2 / 2 - GM_EARTH / r - OMEGA**2 * rho2 / 2
    assert np.ptp(jac) < 1e-6  # m^2/s^2, against |jac| ~ 3e7


def test_sc_multi_is_shuffled_and_split_cleanly(sc_multi):
    assert sc_multi.n_unique_ids == 3
    assert not (np.diff(sc_multi.sspice_gps) >= 0).all()  # really shuffled
    parts = sc_multi.split_by_id()
    assert {str(k) for k in parts} == {"SAT-A", "SAT-B", "SAT-C"}
    assert sum(p.N for p in parts.values()) == sc_multi.N


def test_fake_provider_shape_and_counting():
    prov = FakeProvider()
    out = prov(["SUN", "MOON"], ephem_time=np.arange(5.0))
    assert out.shape == (2, 5, 3) and prov.calls == 1
    np.testing.assert_allclose(np.linalg.norm(out[0], axis=1), 1.495978707e11)


def test_reference_data_is_aligned_and_in_expected_units(
    esa_pod_small, orekit_ref_small
):
    assert len(esa_pod_small) == len(orekit_ref_small) == 1104
    assert (esa_pod_small["DateTime"] == orekit_ref_small["DateTime"]).all()
    r_km = np.linalg.norm(esa_pod_small[["x", "y", "z"]].to_numpy(), axis=1)
    assert 6700 < r_km.min() and r_km.max() < 7000  # km in the raw file
    assert 5.7e7 < orekit_ref_small["earth_gp"].iloc[0] < 5.9e7  # m^2/s^2


def test_esa_constellation_is_si(esa_constellation):
    sc = esa_constellation["ESA"]
    r = np.linalg.norm(sc.state_ecef[:, :3], axis=1)
    v = np.linalg.norm(sc.state_ecef[:, 3:], axis=1)
    assert 6.7e6 < r.min() and r.max() < 7.0e6
    assert 7.4e3 < v.min() and v.max() < 7.8e3


def test_network_guard_blocks_downloads():
    with pytest.raises(RuntimeError, match="network"):
        utils.dl_file("https://example.invalid/x", "x")
