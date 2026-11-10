"""Tests for the ephemeris providers ``SPICEEphem`` and ``OrekitEphem`` (PR T3).

Rows EP-01, EP-02, EP-03, EP-06 and EP-06b of docs/testing.md.
The ``SolarSystemEnvironment`` rows (EP-01b, EP-04, EP-05) are in
``test_solar_system_env.py``.

Fast-tier tests use the bundled de440s kernels (loaded by the ``spice_offline``
fixture) and the stored Orekit columns of ``orekit_ref_small``; only EP-06b
starts the JVM.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import spiceypy as spice

from contigo.constellation import Constellation
from contigo.ephemeris import spice_ephem
from contigo.ephemeris.spice_ephem import SPICEEphem
from contigo.forces.third_body_acc import ThirdBodyEnv
from contigo.solar_system_ephem import SolarSystemEnvironment

BODIES = ["SUN", "MOON"]
SUN_RANGE = (1.470e11, 1.521e11)  # perihelion/aphelion distance, m (see EP-01)
MOON_RANGE = (3.56e8, 4.07e8)  # lunar perigee/apogee distance, m
ONE_KM = 1000.0
# Position tolerance per body (m). SPICE ITRF93 and Orekit's ITRF differ in
# orientation by ~6 mas (3e-8 rad), a *rotation*: measured max 11.4 m at the Moon
# but 4562.5 m at the Sun. The plan's flat 1 km is therefore met by the Moon only;
# the Sun cap of 10 km is ~2.2x the measured miss (the Moon cap is ~88x).
POS_TOL = {"SUN": 10 * ONE_KM, "MOON": ONE_KM}
GPS_UTC_LEAP_2018 = pd.Timedelta(seconds=18)  # GPS - UTC in 2018


def _esa(constellation: Constellation):
    return constellation.spacecraft["ESA"]


def _orekit_positions(ref: pd.DataFrame) -> np.ndarray:
    """Stored Orekit Sun/Moon ECEF positions (m) as a (2, N, 3) array."""
    sun = ref[["ecef_sn_px", "ecef_sn_py", "ecef_sn_pz"]].to_numpy()
    moon = ref[["ecef_mn_px", "ecef_mn_py", "ecef_mn_pz"]].to_numpy()
    return np.stack([sun, moon])


def _assert_epochs_aligned(sc, ref: pd.DataFrame) -> None:
    """Spacecraft epochs equal the reference ``DateTime`` rows (GPS = UTC + 18 s)."""
    gps = pd.DatetimeIndex(sc.sc_utc).tz_localize(None) + GPS_UTC_LEAP_2018
    assert len(gps) == len(ref)
    assert (gps == pd.DatetimeIndex(ref["DateTime"])).all()


def test_ep01_spice_sun_moon_shape_and_range(esa_constellation):
    """EP-01: SPICEEphem Sun/Moon have shape (B,N,3), in metres, at known distances.

    Expected ranges are the physical Sun-Earth (perihelion 1.4710e11 m to
    aphelion 1.5210e11 m) and Earth-Moon (3.56e8 to 4.07e8 m) distance limits.
    The lower Sun bound is 1.470e11 rather than the plan's 1.471e11: the ESA
    epochs (2 Jan 2018) are at perihelion, where the distance is 1.47098e11 m.
    """
    sc = _esa(esa_constellation)
    ephem = SPICEEphem(ephemeris="de440s", frame="ITRF93", observer="EARTH")

    r = ephem(BODIES, ephem_time=sc.sspice_et)

    assert r.shape == (2, len(sc.sspice_et), 3)
    d_sun = np.linalg.norm(r[0], axis=1)
    d_moon = np.linalg.norm(r[1], axis=1)
    assert d_sun.min() > SUN_RANGE[0] and d_sun.max() < SUN_RANGE[1]
    assert d_moon.min() > MOON_RANGE[0] and d_moon.max() < MOON_RANGE[1]


def test_ep01_lowercase_body_names(esa_constellation):
    """EP-01: body names are case-insensitive (``bd.upper()`` in the provider).

    Expected: lower-case names give the same array as upper-case names.
    """
    et = _esa(esa_constellation).sspice_et[:20]
    ephem = SPICEEphem()
    assert np.array_equal(
        ephem(["sun", "moon"], ephem_time=et), ephem(BODIES, ephem_time=et)
    )


def test_ep02_duplicate_epochs_identical_rows(esa_constellation):
    """EP-02: repeated and unsorted epochs return identical rows (unique/inverse).

    Expected: row ``i`` of the output equals the single-epoch result for
    ``et[i]``, and rows with the same epoch are equal.
    """
    et0 = _esa(esa_constellation).sspice_et[:3]
    et = np.array([et0[2], et0[0], et0[2], et0[2], et0[1], et0[0]])
    ephem = SPICEEphem()

    r = ephem(BODIES, ephem_time=et)

    assert r.shape == (2, 6, 3)
    assert np.array_equal(r[:, 0], r[:, 2])
    assert np.array_equal(r[:, 0], r[:, 3])
    assert np.array_equal(r[:, 1], r[:, 5])
    for i, t in enumerate(et):
        single = ephem(BODIES, ephem_time=np.array([t]))
        assert np.array_equal(r[:, i], single[:, 0])


def test_ep03_load_kernels_twice_no_duplicate_furnsh(monkeypatch):
    """EP-03: calling ``load_kernels`` again does not furnish kernels twice.

    Expected: ``spice.ktotal('ALL')`` is unchanged and ``furnsh`` is never called,
    because the ``spice_offline`` fixture already loaded the three kernels.
    """
    n0 = spice.ktotal("ALL")
    ephem = SPICEEphem()
    assert spice.ktotal("ALL") == n0

    calls = []
    monkeypatch.setattr(spice_ephem.spice, "furnsh", lambda fp: calls.append(fp))
    ephem.load_kernels()
    ephem.load_kernels()

    assert calls == []
    assert spice.ktotal("ALL") == n0


def test_ep06_spice_vs_stored_orekit_positions(esa_constellation, orekit_ref_small):
    """EP-06: SPICEEphem Sun/Moon agree with the stored Orekit ECEF positions.

    Expected values are the ``ecef_sn_p*`` / ``ecef_mn_p*`` columns of
    ``orekit_ref_small`` (Orekit, metres) at the same epochs. The plan asks for
    ~1 km. Measured: Moon max 11.4 m, Sun max 4562.5 m. The Sun miss is not an
    ephemeris error: the distances agree to 6 mm and the offset is a 3e-8 rad
    (~6 mas) frame-orientation difference between SPICE ITRF93 and Orekit's ITRF,
    which scales with distance. So: Moon < 1 km, Sun < 10 km (about 2.2x the
    measured 4562.5 m), and the radial (distance) difference < 1 m for both
    bodies. Epochs are checked to line up exactly before comparing rows.
    """
    sc = _esa(esa_constellation)
    _assert_epochs_aligned(sc, orekit_ref_small)
    r_spice = SPICEEphem()(BODIES, ephem_time=sc.sspice_et)
    r_orekit = _orekit_positions(orekit_ref_small)

    assert r_spice.shape == r_orekit.shape
    diff = np.linalg.norm(r_spice - r_orekit, axis=2)  # (B, N), metres
    radial = np.linalg.norm(r_spice, axis=2) - np.linalg.norm(r_orekit, axis=2)
    for i, body in enumerate(BODIES):
        assert diff[i].max() < POS_TOL[body], f"{body} max diff {diff[i].max():.3f} m"
        assert np.abs(radial[i]).max() < 1.0, f"{body} radial diff"


@pytest.mark.orekit
def test_ep06b_orekit_ephem_vs_spice(esa_constellation, orekit_ref_small):
    """EP-06b: live ``OrekitEphem`` matches ``SPICEEphem`` and gives the same
    ``ThirdBodyEnv`` accelerations.

    Expected: positions within ``POS_TOL`` (Moon 1 km, Sun 10 km, see EP-06);
    accelerations agree to a relative vector-norm difference < 1e-6 (the
    notebook saw ~2e-8 relative).
    """
    from contigo.ephemeris.orekit_ephem import OrekitEphem

    sc = _esa(esa_constellation)
    _assert_epochs_aligned(sc, orekit_ref_small)

    # Environments are preloaded with the spacecraft epochs, so each provider
    # is called once; positions are read back through get_ephem and the very
    # same environments then feed ThirdBodyEnv (no second Orekit evaluation).
    envs = {
        name: SolarSystemEnvironment(
            BODIES,
            tolerance=0.001,
            provider=prov,
            ephem_time=sc.sspice_et,
            gps_time=sc.sspice_gps,
            utc_time=sc.sc_utc,
        )
        for name, prov in (("spice", SPICEEphem()), ("orekit", OrekitEphem()))
    }
    r = {
        name: env.get_ephem(
            ephem_time=sc.sspice_et, gps_time=sc.sspice_gps, utc_time=sc.sc_utc
        )[3]
        for name, env in envs.items()
    }
    assert r["orekit"].shape == r["spice"].shape == (2, len(sc.sspice_et), 3)
    diff = np.linalg.norm(r["orekit"] - r["spice"], axis=2)
    for i, body in enumerate(BODIES):
        assert diff[i].max() < POS_TOL[body], f"{body} max diff {diff[i].max():.3f} m"

    acc = {
        name: ThirdBodyEnv().acceleration(esa_constellation, env)["ESA"]
        for name, env in envs.items()
    }
    # Compare vector norms, not components: a component near zero has a huge
    # *relative* error from the ~3e-8 rad frame offset (elementwise rtol fails
    # for ~2% of components, max 6e-5, with absolute errors < 5e-14 m/s^2).
    a_o, a_s = acc["orekit"], acc["spice"]
    rel = np.linalg.norm(a_o - a_s, axis=-1) / np.linalg.norm(a_s, axis=-1)
    assert rel.max() < 1e-6, f"max relative acceleration difference {rel.max():.2e}"
