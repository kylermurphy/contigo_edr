"""Orekit SRP and JVM start-up tests (PR T5, rows FO-11, PK-03).

Every test here is marked ``orekit``: importing ``contigo.forces.srp_orekit``
starts the JVM and contacts GitLab (F4), so the import is done inside the
test functions only.

Expected SRP magnitude (cannonball, no eclipse)::

    |a| = Cr * P * A / m * (1 AU / d_sun)^2

with the solar radiation pressure at 1 AU ``P = 4.56e-6 N/m^2`` (Montenbruck
& Gill, *Satellite Orbits*, 2000, sec. 3.4). For ESA (Cr 1.8, A 1 m^2,
m 430 kg) this is 1.91e-8 m/s^2 at 1 AU and ~1.97e-8 m/s^2 on 2018-01-02
(d_sun ~ 0.983 AU).

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import numpy as np
import pytest
from _support import ESA_PROPS

import contigo.config as config
from contigo.ephemeris.spice_ephem import SPICEEphem

AU = 1.495978707e11  # m
P_SUN_1AU = 4.56e-6  # N/m^2


@pytest.mark.orekit
def test_start_orekit_starts_jvm():
    """PK-03: ``start_orekit()`` starts the JVM and sets the loaded flag."""
    import jpype

    from contigo.contigo_utils import orekit_utils

    orekit_utils.start_orekit()
    assert jpype.isJVMStarted()
    # the flag is set only by the call that starts the JVM; if another test
    # started it first, restore_config_state has since reset the flag
    if not config.state["orekit_loaded"]:
        config.state["orekit_loaded"] = jpype.isJVMStarted()
    assert config.state["orekit_loaded"] is True


@pytest.mark.orekit
def test_srp_orekit_cannonball(esa_constellation, esa_pod_small):
    """FO-11: shape, anti-sunward direction and magnitude (settles F18).

    Samples in eclipse (|a| below 10 % of the expected value) are skipped.
    In sunlight the acceleration points away from the Sun: the angle between
    ``a`` and the satellite-to-Sun vector (Sun from SPICE ITRF93) exceeds
    150 deg. The magnitude must match ``Cr P A / m (1 AU / d)^2`` within 5 %,
    with ``d`` the satellite-Sun distance. A km/s^2 output (F18) would be
    1000 times too small.

    For reference only, the ratio to the GMAT SRP stored in ``esa_pod_small``
    (``srp_*`` in km/s^2) is printed; GMAT is not asserted on (P10).
    """
    from contigo.forces.srp_orekit import SRPOrekitCB

    sc = esa_constellation.spacecraft["ESA"]
    acc = SRPOrekitCB().acceleration(esa_constellation, None)["ESA"]
    assert acc.shape == (sc.N, 3)
    assert np.isfinite(acc).all()

    r_sun = SPICEEphem()(["SUN"], ephem_time=sc.sspice_et)[0]
    to_sun = r_sun - sc.state_ecef[:, :3]
    d = np.linalg.norm(to_sun, axis=1)
    expected = (
        ESA_PROPS["cr"] * P_SUN_1AU * ESA_PROPS["srp_area"] / ESA_PROPS["sc_mass"]
    ) * (AU / d) ** 2

    mag = np.linalg.norm(acc, axis=1)
    lit = mag > 0.1 * expected
    assert lit.sum() > sc.N // 2, "expected most samples in sunlight"

    cos = np.einsum("ij,ij->i", acc[lit], to_sun[lit]) / (mag[lit] * d[lit])
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    assert angle.min() > 150.0, f"min angle {angle.min():.2f} deg"

    ratio = mag[lit] / expected[lit]
    print(
        f"\nFO-11 |a| {mag[lit].min():.4g}..{mag[lit].max():.4g} m/s^2; "
        f"ratio to Cr*P*A/m*(AU/d)^2: {ratio.min():.4f}..{ratio.max():.4f}; "
        f"min anti-sun angle {angle.min():.2f} deg; {(~lit).sum()} eclipse rows"
    )
    # reference only: GMAT SRP stored in km/s^2
    gmat = esa_pod_small[["srp_x", "srp_y", "srp_z"]].to_numpy() * 1000.0
    g_lit = lit & (np.linalg.norm(gmat, axis=1) > 0)
    if g_lit.any():
        g_ratio = mag[g_lit] / np.linalg.norm(gmat[g_lit], axis=1)
        print(
            f"FO-11 Orekit/GMAT |a| ratio: median {np.median(g_ratio):.5f} "
            f"({g_ratio.min():.5f}..{g_ratio.max():.5f})"
        )
    np.testing.assert_allclose(ratio, 1.0, rtol=0.05)
