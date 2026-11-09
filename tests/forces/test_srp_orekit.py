"""Orekit SRP and JVM start-up tests (PR T5, rows FO-11, PK-03).

Every test here is marked ``orekit`` **and** ``network``: importing
``contigo.forces.srp_orekit`` or calling ``start_orekit()`` starts the JVM,
queries GitLab for the latest Orekit data commit and may download the data
(F4). Imports are therefore done inside the test functions only.

Side effect (F12, fixed by R5): when GitLab reports a newer data commit,
``start_orekit()`` deletes the tracked ``contigo/data/orekit_data_*.zip`` and
downloads a new one, leaving the working tree dirty. After running these
tests, restore it with ``git checkout -- contigo/data`` and delete the new
untracked zip.

Expected SRP magnitude (cannonball, no eclipse)::

    |a| = Cr * P * A / m * (1 AU / d_sun)^2

with the solar radiation pressure at 1 AU ``P = 4.56e-6 N/m^2`` (Montenbruck
& Gill, *Satellite Orbits*, 2000, sec. 3.4). For ESA (Cr 1.8, A 1 m^2,
m 430 kg) this is 1.91e-8 m/s^2 at 1 AU and ~1.97e-8 m/s^2 on 2018-01-02
(d_sun ~ 0.983 AU).

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _support import ESA_PROPS

import contigo
from contigo.ephemeris.spice_ephem import SPICEEphem

AU = 1.495978707e11  # m
P_SUN_1AU = 4.56e-6  # N/m^2

PK03_SCRIPT = """
import jpype
import contigo.config as config
from contigo.contigo_utils import orekit_utils
orekit_utils.start_orekit()
print("PK03", jpype.isJVMStarted(), config.state["orekit_loaded"], flush=True)
"""


@pytest.mark.orekit
@pytest.mark.network
def test_start_orekit_starts_jvm(tmp_path):
    """PK-03: ``start_orekit()`` starts the JVM and sets the loaded flag.

    Runs in a fresh interpreter (cwd ``tmp_path``, ``PYTHONPATH`` pointing at
    the contigo copy under test), so neither a JVM started by another test
    nor ``restore_config_state`` can mask the result.
    """
    pkg_root = str(Path(contigo.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (pkg_root, env.get("PYTHONPATH")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-c", PK03_SCRIPT],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PK03")]
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert lines == ["PK03 True True"], proc.stdout[-2000:]


@pytest.mark.orekit
@pytest.mark.network
def test_srp_orekit_cannonball(esa_constellation, esa_pod_small):
    """FO-11: shape, anti-sunward direction and magnitude (settles F18).

    Only nearly fully lit samples (|a| > 98 % of the expected value) are
    checked. These two orbits contain no eclipse (GMAT's stored SRP shows
    none either), so the eclipse branch is not exercised; at least 90 % of
    the samples must pass the mask so the test cannot pass vacuously.
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
    lit = mag > 0.98 * expected
    assert lit.sum() > 0.9 * sc.N, f"only {lit.sum()} of {sc.N} samples fully lit"

    cos = np.einsum("ij,ij->i", acc[lit], to_sun[lit]) / (mag[lit] * d[lit])
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    assert angle.min() > 150.0, f"min angle {angle.min():.2f} deg"

    ratio = mag[lit] / expected[lit]
    print(
        f"\nFO-11 |a| {mag[lit].min():.4g}..{mag[lit].max():.4g} m/s^2; "
        f"ratio to Cr*P*A/m*(AU/d)^2: {ratio.min():.6f}..{ratio.max():.6f}; "
        f"min anti-sun angle {angle.min():.2f} deg; {(~lit).sum()} masked rows"
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
