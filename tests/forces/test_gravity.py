"""Tests for the Earth gravity potential (PR T4).

Covers ``contigo.forces.grav_utils`` (ICGEM reader, scalar and batched
spherical-harmonic potential), ``GravPot`` and the ``EarthPotential`` force
model. Rows FO-04 to FO-08 and FO-13 of docs/phase0_plan.md §4.5.

Normalisation: pyshtools ``PlmBar`` uses 4pi ("geodesy") normalisation without
the Condon-Shortley phase,
``Pbar_lm = sqrt((2 - delta_m0) (2l + 1) (l - m)! / (l + m)!) P_lm``, evaluated
at ``x = cos(colatitude) = sin(latitude)`` (geocentric latitude).

added: 09/11/2026 Claude Code
"""

import os

import numpy as np
import pytest

import contigo.config as config
from contigo.forces.grav_pot import EarthPotential, GravPot
from contigo.forces.grav_utils import (
    get_potential,
    get_potential_batch,
    read_icgem_coeff,
)

EIGEN2 = os.path.join(config.DATA_DIR, "EIGEN-2.gfc")

# Header values of contigo/data/EIGEN-2.gfc
EIGEN2_GM = 0.3986004415e15
EIGEN2_R0 = 0.6378136460e07
EIGEN2_LMAX = 140
EIGEN2_C20 = -0.484165815935e-03
EIGEN2_C22 = 0.243930491835e-05
EIGEN2_S22 = -0.140027520233e-05

# Test points: radii from LEO to GEO, latitudes covering poles and equator
R_PTS = np.array([6_678_137.0, 6_771_000.0, 7_078_137.0, 4.2164e7, 6_900_000.0])
LAT_PTS = np.radians([0.0, 45.0, -60.0, 89.9, -90.0])
LON_PTS = np.radians([0.0, 123.4, -170.0, 33.0, 270.0])


@pytest.fixture(scope="module")
def eigen2():
    return read_icgem_coeff(EIGEN2)


def _zero_coeffs(lmax):
    clm = np.zeros((lmax + 1, lmax + 1))
    slm = np.zeros((lmax + 1, lmax + 1))
    clm[0, 0] = 1.0
    return clm, slm


# ---------------------------------------------------------------------------
# FO-04 read_icgem_coeff
# ---------------------------------------------------------------------------
def test_read_icgem_eigen2_header_and_coefficients(eigen2):
    """FO-04: EIGEN-2 header and selected coefficients match the file.

    Expected values are copied from the header and the ``gfc`` lines of
    contigo/data/EIGEN-2.gfc (C20 = -.484165815935e-03,
    C22/S22 = 0.243930491835e-05 / -.140027520233e-05).
    """
    clm, slm, meta = eigen2
    assert meta["lmax"] == EIGEN2_LMAX
    assert meta["GM"] == EIGEN2_GM == 3.986004415e14
    assert meta["r0"] == EIGEN2_R0 == 6378136.46
    assert meta["product"] == "gravity_field"
    assert clm.shape == slm.shape == (EIGEN2_LMAX + 1, EIGEN2_LMAX + 1)
    assert clm[0, 0] == 1.0
    assert clm[2, 0] == EIGEN2_C20
    assert clm[2, 0] == pytest.approx(-4.8417e-4, abs=5e-9)
    assert clm[2, 2] == EIGEN2_C22
    assert slm[2, 2] == EIGEN2_S22
    # S_l0 is identically zero (sin(0 * lon) = 0); no coefficients above m = l
    np.testing.assert_array_equal(slm[:, 0], 0.0)
    assert np.all(np.triu(clm, k=1) == 0.0)
    assert np.all(np.triu(slm, k=1) == 0.0)


def test_read_icgem_no_max_degree_raises(tmp_path):
    """FO-04: gfc data lines without a preceding max_degree header -> ValueError."""
    f = tmp_path / "nomax.gfc"
    f.write_text(
        "product_type gravity_field\n"
        "earth_gravity_constant 0.3986004415E+15\n"
        "radius 0.6378136460E+07\n"
        "end_of_head ====\n"
        "gfc 0 0 0.1e+01 0.0e+00 0.0 0.0\n"
    )
    with pytest.raises(ValueError):
        read_icgem_coeff(str(f))


def test_read_icgem_missing_radius_raises(tmp_path):
    """FO-04: a header without ``radius`` -> ValueError."""
    f = tmp_path / "norad.gfc"
    f.write_text(
        "product_type gravity_field\n"
        "earth_gravity_constant 0.3986004415E+15\n"
        "max_degree 2\n"
        "end_of_head ====\n"
        "gfc 0 0 0.1e+01 0.0e+00 0.0 0.0\n"
        "gfc 2 0 -.484165815935e-03 0.0e+00 0.0 0.0\n"
    )
    with pytest.raises(ValueError, match="missing required header"):
        read_icgem_coeff(str(f))


# ---------------------------------------------------------------------------
# FO-05 get_potential_batch against closed forms
# ---------------------------------------------------------------------------
def test_batch_lmax0_is_point_mass():
    """FO-05: lmax=0 gives the point-mass potential U = GM/r (rtol 1e-14)."""
    clm, slm = _zero_coeffs(0)
    u = get_potential_batch(
        R_PTS, LAT_PTS, LON_PTS, clm, slm, EIGEN2_GM, EIGEN2_R0, lmax=0
    )
    np.testing.assert_allclose(u, EIGEN2_GM / R_PTS, rtol=1e-14)


def test_batch_lmax2_zonal_c20_closed_form():
    """FO-05: lmax=2 with only C00=1 and C20 equals the J2 closed form.

    ``U = GM/r [1 + (r0/r)^2 C20 Pbar_20(sin lat)]`` with
    ``Pbar_20(x) = sqrt(5) (3 x^2 - 1) / 2`` (4pi normalisation, see module
    docstring). C20 is the EIGEN-2 value.
    """
    clm, slm = _zero_coeffs(2)
    clm[2, 0] = EIGEN2_C20
    u = get_potential_batch(
        R_PTS, LAT_PTS, LON_PTS, clm, slm, EIGEN2_GM, EIGEN2_R0, lmax=2
    )
    x = np.sin(LAT_PTS)
    p20 = np.sqrt(5.0) * (3.0 * x**2 - 1.0) / 2.0
    expected = EIGEN2_GM / R_PTS * (1.0 + (EIGEN2_R0 / R_PTS) ** 2 * EIGEN2_C20 * p20)
    np.testing.assert_allclose(u, expected, rtol=1e-12)


def test_batch_lmax2_tesseral_sectoral_closed_form():
    """FO-05: lmax=2 with C21/S21 and C22/S22 equals the closed form.

    Without the Condon-Shortley phase ``P_21 = 3 x sqrt(1 - x^2)`` and
    ``P_22 = 3 (1 - x^2)``; with the 4pi factors sqrt(10/6) and sqrt(10/24):
    ``Pbar_21(x) = sqrt(15) x sqrt(1 - x^2)`` and
    ``Pbar_22(x) = sqrt(15)/2 (1 - x^2)``. With ``x = sin(lat)``:
    ``U = GM/r [1 + (r0/r)^2 (Pbar_21 (C21 cos l + S21 sin l)
    + Pbar_22 (C22 cos 2l + S22 sin 2l))]``.
    C22/S22 are the EIGEN-2 values; C21/S21 are arbitrary non-zero values.
    """
    c21, s21 = 3.0e-6, -2.0e-6
    clm, slm = _zero_coeffs(2)
    clm[2, 1], slm[2, 1] = c21, s21
    clm[2, 2], slm[2, 2] = EIGEN2_C22, EIGEN2_S22
    u = get_potential_batch(
        R_PTS, LAT_PTS, LON_PTS, clm, slm, EIGEN2_GM, EIGEN2_R0, lmax=2
    )
    x = np.sin(LAT_PTS)
    p21 = np.sqrt(15.0) * x * np.sqrt(1.0 - x**2)
    p22 = np.sqrt(15.0) / 2.0 * (1.0 - x**2)
    lam = LON_PTS
    harm = p21 * (c21 * np.cos(lam) + s21 * np.sin(lam)) + p22 * (
        EIGEN2_C22 * np.cos(2 * lam) + EIGEN2_S22 * np.sin(2 * lam)
    )
    expected = EIGEN2_GM / R_PTS * (1.0 + (EIGEN2_R0 / R_PTS) ** 2 * harm)
    np.testing.assert_allclose(u, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# FO-06 scalar vs batch
# ---------------------------------------------------------------------------
def test_scalar_matches_batch(eigen2):
    """FO-06: ``get_potential`` equals ``get_potential_batch`` (EIGEN-2, lmax 50)."""
    clm, slm, meta = eigen2
    batch = get_potential_batch(
        R_PTS, LAT_PTS, LON_PTS, clm, slm, meta["GM"], meta["r0"], lmax=50
    )
    scalar = np.array(
        [
            get_potential(r, la, lo, clm, slm, meta["GM"], meta["r0"], lmax=50)
            for r, la, lo in zip(R_PTS, LAT_PTS, LON_PTS, strict=True)
        ]
    )
    np.testing.assert_allclose(scalar, batch, rtol=1e-13)


# ---------------------------------------------------------------------------
# FO-07 GravPot
# ---------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason="F7/R7: get_pot returns None before calc_pot")
def test_gravpot_get_pot_before_calc_raises():
    """FO-07: ``get_pot()`` before ``calc_pot()`` raises ValueError."""
    gp = GravPot(pot_file="EIGEN-2.gfc", lmax=2)
    with pytest.raises(ValueError):
        gp.get_pot()


def test_gravpot_lmax_clamped_on_first_load():
    """FO-07: lmax above the file maximum is clamped to 140 when the file is read."""
    config.state["pot_coef_loaded"] = False
    config.state["pot_file"] = None
    gp = GravPot(pot_file="EIGEN-2.gfc", lmax=500)
    assert gp.lmax == EIGEN2_LMAX
    assert config.state["pot_coef_loaded"] is True
    assert config.state["pot_file"] == "EIGEN-2.gfc"


@pytest.mark.xfail(strict=True, reason="F7/R7: lmax not clamped on cached path")
def test_gravpot_lmax_clamped_on_cached_path():
    """FO-07: lmax is also clamped when coefficients come from config.state.

    The first GravPot populates the cache; the second must take the cached
    branch (checked via identity of ``clm``) and still clamp lmax to 140.
    ``calc_pot`` is not called: with lmax 500 the Numba kernel would index
    beyond the (141, 141) coefficient arrays without bounds checking.
    """
    config.state["pot_coef_loaded"] = False
    config.state["pot_file"] = None
    first = GravPot(pot_file="EIGEN-2.gfc", lmax=20)
    second = GravPot(pot_file="EIGEN-2.gfc", lmax=500)
    assert second.clm is first.clm  # cached branch was taken
    assert second.lmax == EIGEN2_LMAX


def test_gravpot_missing_file_raises(tmp_path):
    """FO-07: a potential file that exists nowhere -> ValueError."""
    with pytest.raises(ValueError, match="cannot be found"):
        GravPot(pot_file=str(tmp_path / "no_such_model.gfc"))


# ---------------------------------------------------------------------------
# FO-08 EarthPotential
# ---------------------------------------------------------------------------
def test_earth_potential_dict_shapes(constellation_fixture):
    """FO-08: ``potential`` returns {id: (N,) float array} close to GM/r.

    Non-central terms are at most ~1e-3 of GM/r (J2 dominated), so the
    values are compared with GM/r at rtol 2e-3.
    """
    ep = EarthPotential(pot_file="EIGEN-2.gfc", lmax=20)
    out = ep.potential(constellation_fixture)
    assert set(out) == set(constellation_fixture.spacecraft)
    for sc_id, sc in constellation_fixture.spacecraft.items():
        u = out[sc_id]
        assert isinstance(u, np.ndarray)
        assert u.dtype == np.float64
        assert u.shape == (sc.N,)
        assert np.all(np.isfinite(u)) and np.all(u > 0)
        r = np.linalg.norm(sc.state_ecef[:, :3], axis=1)
        np.testing.assert_allclose(u, EIGEN2_GM / r, rtol=2e-3)


def test_earth_potential_acceleration_not_implemented(constellation_fixture):
    """FO-08: ``acceleration`` raises NotImplementedError."""
    ep = EarthPotential(pot_file="EIGEN-2.gfc", lmax=20)
    with pytest.raises(NotImplementedError):
        ep.acceleration(constellation_fixture, None)


# ---------------------------------------------------------------------------
# FO-13 EarthPotential vs Orekit
# ---------------------------------------------------------------------------
def test_earth_potential_matches_orekit(esa_constellation, orekit_ref_small):
    """FO-13: EIGEN-2 lmax 100 on the ESA POD matches Orekit ``earth_gp``.

    Reference: Orekit potential stored in ``orekit_ref_small['earth_gp']``
    (m^2/s^2) for the same epochs (tests/data/make_fixtures.py). Notebooks
    showed agreement at ~1e-9 relative, but that was display rounding in
    km^2/s^2. Orekit (Orekit_EDR.ipynb: HolmesFeatherstoneAttractionModel,
    EIGEN-2, degree/order 100) and CONTIGO agree to ~3e-16 relative (measured),
    so besides the ``np.allclose`` default a rtol of 1e-12 is also asserted.
    """
    ep = EarthPotential(pot_file="EIGEN-2.gfc", lmax=100)
    u = ep.potential(esa_constellation)["ESA"]
    ref = orekit_ref_small["earth_gp"].to_numpy()
    assert u.shape == ref.shape
    assert np.allclose(u, ref)
    np.testing.assert_allclose(u, ref, rtol=1e-12, atol=0)
