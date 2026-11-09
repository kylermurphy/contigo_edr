"""Tests for ``contigo.spacecraft.Spacecraft`` (PR T1, catalogue §4.1).

Rows SC-01 to SC-12 (plus SC-04b and SC-11b) of ``docs/phase0_plan.md``. Every
test docstring starts with its catalogue row ID and states where the expected
numbers come from. Files are written to ``tmp_path``; nothing needs the network,
the JVM or GMAT.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import bz2
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _support import ESA_PROPS, MULTI_MEMBERS, make_circular_orbit

from contigo.contigo_utils.utils import df_sp3
from contigo.spacecraft import Spacecraft

#: TDB - GPS offset at J2000-era epochs: TDB - TAI = 32.184 s (+ periodic term
#: below 2 ms), TAI - GPS = 19 s.
ET_MINUS_GPS = 51.184

STATE_COLS = ["x", "y", "z", "vx", "vy", "vz"]


def _frame(n: int = 20, **orbit_kwargs) -> pd.DataFrame:
    """DataFrame (time, x..vz) from the exact circular orbit, SI units."""
    state, time = make_circular_orbit(n=n, **orbit_kwargs)
    df = pd.DataFrame(state, columns=STATE_COLS)
    df.insert(0, "time", time)
    return df


# ---------------------------------------------------------------------------
# SC-01 .. SC-04b: construction from arrays
# ---------------------------------------------------------------------------
def test_sc01_load_from_arrays(sc_single):
    """SC-01: N, shapes and the default ``NO_ID``.

    Expected: ``sc_single`` is built from 600 samples (conftest), so every
    per-sample array has 600 entries and the state is (600, 6).
    """
    sc = sc_single
    assert sc.N == 600
    assert sc.state_ecef.shape == (600, 6)
    assert len(sc.stime) == 600
    for arr in (sc.sspice_et, sc.sspice_gps, sc.sc_utc, sc.sc_id):
        assert arr.shape == (600,)
    assert list(sc.unique_ids) == ["SAT-A"]
    assert sc.n_unique_ids == 1
    assert repr(sc) == "Spacecraft(N=600, n_unique_ids=1)"

    # without sc_id_input the ID defaults to "NO_ID"
    state, time = make_circular_orbit(n=12)
    anon = Spacecraft(state=state, time=time, tscale_input="GPS")
    assert anon.N == 12
    assert list(anon.unique_ids) == ["NO_ID"]
    assert set(anon.sc_id) == {"NO_ID"}


def test_sc02_unit_input_km():
    """SC-02: ``unit_input='km'`` multiplies positions and velocities by 1000.

    Expected: the orbit is generated in m and m/s; the same numbers divided by
    1000 are passed in as km and km/s, so the result must equal the original.
    """
    state, time = make_circular_orbit(n=30)
    sc = Spacecraft(
        state=state / 1000.0, time=time, tscale_input="GPS", unit_input="km"
    )
    np.testing.assert_allclose(sc.state_ecef, state, rtol=1e-14)
    # a km spacecraft sits ~6778 km from the centre, 6.778e6 m internally
    assert 6.7e6 < np.linalg.norm(sc.state_ecef[0, :3]) < 6.8e6

    sc_m = Spacecraft(state=state, time=time, tscale_input="GPS", unit_input="m")
    np.testing.assert_array_equal(sc_m.state_ecef, state)


def test_sc02_km_does_not_mutate_input():
    """SC-02 (extra): the caller's array is left untouched by the km conversion.

    Expected: a copy taken before construction equals the array afterwards.
    """
    state, time = make_circular_orbit(n=10)
    state_km = state / 1000.0
    before = state_km.copy()
    Spacecraft(state=state_km, time=time, tscale_input="GPS", unit_input="km")
    np.testing.assert_array_equal(state_km, before)


def test_sc02_m_does_not_share_input():
    """SC-02 (extra): with ``unit_input='m'`` the object owns its state too.

    Expected: the internal array is not the caller's array (nor a view of it),
    so editing either one leaves the other unchanged.
    """
    state, time = make_circular_orbit(n=10)
    before = state.copy()
    sc = Spacecraft(state=state, time=time, tscale_input="GPS", unit_input="m")
    assert not np.shares_memory(sc.state_ecef, state)
    sc.state_ecef[0, 0] = -1.0
    np.testing.assert_array_equal(state, before)
    state[1, 1] = -2.0
    assert sc.state_ecef[1, 1] == before[1, 1]


def test_sc02_km_reload_converts_once():
    """SC-02 (extra): reloading a km spacecraft converts units exactly once.

    Expected: loading the same km array twice leaves the state at x1000 of
    the input (the original SI orbit), not x1e6, and the input is unchanged.
    """
    state, time = make_circular_orbit(n=10)
    state_km = state / 1000.0
    before = state_km.copy()
    sc = Spacecraft(state=state_km, time=time, tscale_input="GPS", unit_input="km")
    sc.load_from_arrays(state_km, time)
    np.testing.assert_allclose(sc.state_ecef, state, rtol=1e-14)
    sc.load_from_arrays(state_km, time)
    np.testing.assert_allclose(sc.state_ecef, state, rtol=1e-14)
    np.testing.assert_array_equal(state_km, before)


_BAD_STATE, _BAD_TIME = make_circular_orbit(n=10)


def _build_bad_inputs(state, time):
    return {
        "no_tscale": dict(state=state, time=time),
        "bad_scale": dict(state=state, time=time, tscale_input="XYZ"),
        "state_n5": dict(state=state[:, :5], time=time, tscale_input="GPS"),
        "state_1d": dict(state=state[:, 0], time=time, tscale_input="GPS"),
        "length_mismatch": dict(state=state, time=time[:7], tscale_input="GPS"),
        "state_without_time": dict(state=state, tscale_input="GPS"),
        "time_without_state": dict(time=time, tscale_input="GPS"),
        "neither_arrays_nor_file": dict(tscale_input="GPS"),
        "bad_unit": dict(
            state=state, time=time, tscale_input="GPS", unit_input="parsec"
        ),
        "sc_id_wrong_length": dict(
            state=state, time=time, tscale_input="GPS", sc_id_input=["A"] * 3
        ),
    }


BAD_INPUTS = _build_bad_inputs(_BAD_STATE, _BAD_TIME)


@pytest.mark.parametrize("case", list(BAD_INPUTS))
def test_sc03_bad_input_raises_value_error(case):
    """SC-03: invalid constructor input raises ``ValueError``.

    Expected: each case is one of the failure modes listed in the catalogue
    (missing/unknown time scale, state not (N,6), length mismatch, state or
    time alone, no input at all) or a sibling check in the same code path.
    """
    with pytest.raises(ValueError):
        Spacecraft(**BAD_INPUTS[case])


def test_sc04_property_broadcasting():
    """SC-04: scalar -> (N,), length-N array kept, wrong length raises, missing NaN.

    Expected: N = 15; ``cd`` scalar becomes 15 copies, ``drag_area`` array is
    returned as given, unspecified properties are all NaN.
    """
    state, time = make_circular_orbit(n=15)
    area = np.linspace(1.0, 2.0, 15)
    sc = Spacecraft(
        state=state, time=time, tscale_input="GPS", cd=2.2, drag_area=area, cr=1.5
    )
    np.testing.assert_array_equal(sc.cd_arr, np.full(15, 2.2))
    np.testing.assert_array_equal(sc.drag_area_arr, area)
    np.testing.assert_array_equal(sc.cr_arr, np.full(15, 1.5))
    assert np.isnan(sc.sc_mass_arr).all() and sc.sc_mass_arr.shape == (15,)
    assert np.isnan(sc.srp_area_arr).all() and sc.srp_area_arr.shape == (15,)

    for bad in (np.ones(14), np.ones(16), np.ones((15, 2))):
        with pytest.raises(ValueError):
            Spacecraft(state=state, time=time, tscale_input="GPS", cd=bad)


def test_sc04b_property_columns_and_kwarg_precedence(tmp_path):
    """SC-04b: property columns are read from the file; a scalar kwarg wins.

    Expected: the CSV carries cd=2.5, drag area 1.1, mass 430, cr 1.8,
    srp area 0.9 (the numbers written below). With ``cd=3.0`` passed
    explicitly, only ``cd`` changes.
    """
    df = _frame(n=8)
    df["Cd"] = 2.5
    df["A_drag"] = np.linspace(1.0, 1.7, 8)
    df["M"] = 430.0
    df["Cr"] = 1.8
    df["A_srp"] = 0.9
    path = tmp_path / "props.csv"
    df.to_csv(path, index=False)
    cols = dict(
        cd_col="Cd",
        drag_area_col="A_drag",
        sc_mass_col="M",
        cr_col="Cr",
        srp_area_col="A_srp",
    )

    sc = Spacecraft(state_file=path, tscale_input="GPS", **cols)
    np.testing.assert_allclose(sc.cd_arr, 2.5)
    np.testing.assert_allclose(sc.drag_area_arr, df["A_drag"])
    np.testing.assert_allclose(sc.sc_mass_arr, 430.0)
    np.testing.assert_allclose(sc.cr_arr, 1.8)
    np.testing.assert_allclose(sc.srp_area_arr, 0.9)

    sc2 = Spacecraft(state_file=path, tscale_input="GPS", cd=3.0, **cols)
    np.testing.assert_array_equal(sc2.cd_arr, np.full(8, 3.0))
    np.testing.assert_allclose(sc2.drag_area_arr, df["A_drag"])
    np.testing.assert_allclose(sc2.cr_arr, 1.8)

    # no column named and no kwarg: NaN
    sc3 = Spacecraft(state_file=path, tscale_input="GPS")
    assert np.isnan(sc3.cd_arr).all()


# ---------------------------------------------------------------------------
# SC-05, SC-06: file loaders
# ---------------------------------------------------------------------------
CUSTOM = dict(
    time_col="t",
    x_col="px",
    y_col="py",
    z_col="pz",
    vx_col="qx",
    vy_col="qy",
    vz_col="qz",
)


def _custom_frame(n: int = 20, **orbit_kwargs) -> pd.DataFrame:
    df = _frame(n, **orbit_kwargs)
    return df.set_axis(["t", "px", "py", "pz", "qx", "qy", "qz"], axis="columns")


def _check_custom(sc: Spacecraft, df: pd.DataFrame) -> None:
    np.testing.assert_allclose(
        sc.state_ecef, df[["px", "py", "pz", "qx", "qy", "qz"]].to_numpy(), rtol=1e-12
    )
    assert (sc.stime == pd.DatetimeIndex(df["t"])).all()


def test_sc05_csv_custom_columns_and_id_input(tmp_path):
    """SC-05: CSV loader with custom column names.

    Expected: the state and time loaded equal the DataFrame that was written;
    ``sc_id_input`` supplies the ID when the file has no ID column.
    """
    df = _custom_frame()
    path = tmp_path / "orbit.csv"
    df.to_csv(path, index=False)
    sc = Spacecraft(state_file=path, tscale_input="GPS", sc_id_input="SAT-X", **CUSTOM)
    assert sc.N == 20
    _check_custom(sc, df)
    assert list(sc.unique_ids) == ["SAT-X"]

    sc_default = Spacecraft(state_file=str(path), tscale_input="GPS", **CUSTOM)
    assert list(sc_default.unique_ids) == ["NO_ID"]


def test_sc05_csv_sc_id_column(tmp_path):
    """SC-05: ``sc_id_col`` reads the spacecraft ID from the file.

    Expected: 10 rows labelled A then 10 labelled B give two IDs of 10 each.
    """
    df = _custom_frame(20)
    df["name"] = ["A"] * 10 + ["B"] * 10
    path = tmp_path / "ids.csv"
    df.to_csv(path, index=False)
    sc = Spacecraft(state_file=path, tscale_input="GPS", sc_id_col="name", **CUSTOM)
    assert list(sc.unique_ids) == ["A", "B"]
    assert (sc.sc_id[:10] == "A").all() and (sc.sc_id[10:] == "B").all()


def test_sc05_hdf_loader(tmp_path):
    """SC-05: HDF loader (``pd.read_hdf``) with default column names.

    Expected: exact round trip of the DataFrame written with ``to_hdf``.
    """
    df = _frame(25)
    df["sat"] = "H-1"
    path = tmp_path / "orbit.hdf"
    df.to_hdf(path, key="orbit", mode="w")
    sc = Spacecraft(state_file=path, tscale_input="GPS", sc_id_col="sat")
    assert sc.N == 25
    np.testing.assert_array_equal(sc.state_ecef, df[STATE_COLS].to_numpy())
    assert (sc.stime == pd.DatetimeIndex(df["time"])).all()
    assert list(sc.unique_ids) == ["H-1"]

    # the h5 alias goes to the same reader
    path5 = tmp_path / "orbit.h5"
    df.to_hdf(path5, key="orbit", mode="w")
    sc5 = Spacecraft(state_file=path5, tscale_input="GPS")
    np.testing.assert_array_equal(sc5.state_ecef, df[STATE_COLS].to_numpy())


def test_sc05_glob_over_two_files_with_filename_id(tmp_path):
    """SC-05: glob over 2 files and ``sc_id_col='filename'`` + ``sc_fn_slc``.

    Expected: files ``SAT-A_day1.csv`` (12 rows) and ``SAT-B_day2.csv`` (8 rows)
    give N = 20 and IDs ``SAT-A``/``SAT-B`` from the first five characters of
    the base file name. The glob result is sorted so SAT-A rows come first.
    """
    dfa, dfb = _frame(12), _frame(8, alt=500e3, inc_deg=87.0)
    dfa.to_csv(tmp_path / "SAT-A_day1.csv", index=False)
    dfb.to_csv(tmp_path / "SAT-B_day2.csv", index=False)
    sc = Spacecraft(
        state_file=str(tmp_path / "*.csv"),
        tscale_input="GPS",
        sc_id_col="filename",
        sc_fn_slc=slice(0, 5),
    )
    assert sc.N == 20
    assert list(sc.unique_ids) == ["SAT-A", "SAT-B"]
    assert (sc.sc_id[:12] == "SAT-A").all() and (sc.sc_id[12:] == "SAT-B").all()
    np.testing.assert_allclose(sc.state_ecef[:12], dfa[STATE_COLS], rtol=1e-12)
    np.testing.assert_allclose(sc.state_ecef[12:], dfb[STATE_COLS], rtol=1e-12)

    # without a slice the whole base name is the ID
    sc_full = Spacecraft(
        state_file=str(tmp_path / "*.csv"), tscale_input="GPS", sc_id_col="filename"
    )
    assert list(sc_full.unique_ids) == ["SAT-A_day1.csv", "SAT-B_day2.csv"]


def test_sc05_zipped_and_mixed_csv(tmp_path):
    """SC-05: ``.csv.zip`` and a list mixing zipped and plain CSVs.

    Expected: pandas infers the zip compression from the extension; the
    explicit list is concatenated in sorted path order (``a_*`` before ``b_*``).
    """
    dfa, dfb = _frame(9), _frame(11, alt=500e3, inc_deg=87.0)
    zipped = tmp_path / "a_orbit.csv.zip"
    plain = tmp_path / "b_orbit.csv"
    dfa.to_csv(zipped, index=False, compression="zip")
    dfb.to_csv(plain, index=False)

    sc_zip = Spacecraft(state_file=zipped, tscale_input="GPS")
    assert sc_zip.N == 9
    np.testing.assert_allclose(sc_zip.state_ecef, dfa[STATE_COLS], rtol=1e-12)

    sc_mix = Spacecraft(state_file=[plain, zipped], tscale_input="GPS")
    assert sc_mix.N == 20
    np.testing.assert_allclose(sc_mix.state_ecef[:9], dfa[STATE_COLS], rtol=1e-12)
    np.testing.assert_allclose(sc_mix.state_ecef[9:], dfb[STATE_COLS], rtol=1e-12)


def test_sc06_loader_errors(tmp_path):
    """SC-06: missing column, no files and unknown extension.

    Expected: a CSV without ``vz`` -> ``ValueError``; a glob that matches
    nothing -> ``FileNotFoundError``; an existing ``.dat`` file -> ``ValueError``
    (unsupported loader).
    """
    df = _frame(5)
    short = tmp_path / "short.csv"
    df.drop(columns="vz").to_csv(short, index=False)
    with pytest.raises(ValueError, match="vz"):
        Spacecraft(state_file=short, tscale_input="GPS")

    with pytest.raises(FileNotFoundError):
        Spacecraft(state_file=str(tmp_path / "nothing_*.csv"), tscale_input="GPS")
    with pytest.raises(FileNotFoundError):
        Spacecraft(state_file=[tmp_path / "missing.csv"], tscale_input="GPS")

    odd = tmp_path / "orbit.dat"
    df.to_csv(odd, index=False)
    with pytest.raises(ValueError, match="Unsupported loader"):
        Spacecraft(state_file=odd, tscale_input="GPS")


# ---------------------------------------------------------------------------
# SC-07: split_by_id
# ---------------------------------------------------------------------------
def test_sc07_split_by_id(sc_multi):
    """SC-07: one object per ID, sorted in time, properties sliced consistently.

    Expected: ``sc_multi`` is three 200-sample orbits (``MULTI_MEMBERS``)
    with rows shuffled. Each member, after splitting, must equal a fresh
    ``make_circular_orbit`` of its altitude/inclination, and ``cd`` must be
    that member's own value. Total N is 600.
    """
    parts = sc_multi.split_by_id()
    assert set(parts) == {m[0] for m in MULTI_MEMBERS}
    assert sum(p.N for p in parts.values()) == sc_multi.N == 600

    # the fixture really is shuffled, so sorting is exercised
    assert not (np.diff(sc_multi.sspice_gps) >= 0).all()

    for sc_id, alt, inc, cd in MULTI_MEMBERS:
        sc = parts[sc_id]
        state, time = make_circular_orbit(alt=alt, inc_deg=inc, n=200)
        assert sc.N == 200
        assert list(sc.unique_ids) == [sc_id]
        np.testing.assert_allclose(np.diff(sc.sspice_gps), 10.0, atol=1e-6)
        assert (sc.stime == time).all()
        np.testing.assert_array_equal(sc.state_ecef, state)
        np.testing.assert_array_equal(sc.cd_arr, np.full(200, cd))
        np.testing.assert_array_equal(sc.drag_area_arr, np.full(200, 1.1))
        np.testing.assert_array_equal(sc.sc_mass_arr, np.full(200, 430.0))
        np.testing.assert_array_equal(sc.cr_arr, np.full(200, 1.8))
        np.testing.assert_array_equal(sc.srp_area_arr, np.full(200, 1.0))


def test_sc07_split_single_id_is_identity(sc_single):
    """SC-07: splitting a one-ID Spacecraft returns one equal object.

    Expected: the sample order of ``sc_single`` is already time-sorted, so the
    state is unchanged.
    """
    parts = sc_single.split_by_id()
    assert list(parts) == ["SAT-A"]
    np.testing.assert_array_equal(parts["SAT-A"].state_ecef, sc_single.state_ecef)
    np.testing.assert_array_equal(parts["SAT-A"].cd_arr, np.full(600, ESA_PROPS["cd"]))


# ---------------------------------------------------------------------------
# SC-08, SC-09: time axes
# ---------------------------------------------------------------------------
def test_sc08_time_axes_gps(sc_single):
    """SC-08: ``|sspice_et - sspice_gps - 51.184| < 2e-3`` s for GPS input.

    Expected: TDB - TAI = 32.184 s plus a periodic term below 2 ms, and
    TAI - GPS = 19 s, so ET - GPS = 51.184 s to within 2 ms.
    """
    dt = sc_single.sspice_et - sc_single.sspice_gps
    assert np.abs(dt - ET_MINUS_GPS).max() < 2e-3
    # the 10 s sampling survives the conversion
    np.testing.assert_allclose(np.diff(sc_single.sspice_gps), 10.0, atol=1e-6)
    assert len(sc_single.sc_utc) == sc_single.N


def test_sc08_utc_axis_matches_leap_seconds(sc_single):
    """SC-08: ``sc_utc`` is UTC, i.e. GPS time minus 18 s in 2018.

    Expected: GPS - UTC = 18 s from 2017-01-01 (leap-second table in
    ``naif0012.tls``); sample 0 is 2018-01-02 00:00:00 GPS = 23:59:42 UTC the
    previous day.
    """
    first = pd.Timestamp(sc_single.sc_utc[0]).tz_localize(None)
    assert first == pd.Timestamp("2018-01-01 23:59:42")


@pytest.mark.parametrize(
    ("utc", "gps", "offset"),
    [
        ("2017-01-01T00:00:00", "2017-01-01T00:00:18", 18),
        ("2005-08-16T00:00:00", "2005-08-16T00:00:13", 13),
    ],
)
def test_sc09_leap_seconds(utc, gps, offset):
    """SC-09: UTC input and GPS input of the same instant agree.

    Expected: GPS - UTC is 18 s since 2017-01-01 and 13 s between 1999-01-01
    and 2006-01-01 (CHAMP era). So a UTC stamp and the GPS stamp ``offset``
    seconds later denote the same instant and must give the same GPS seconds
    and the same ET.
    """
    state = make_circular_orbit(n=3)[0]
    assert (pd.Timestamp(gps) - pd.Timestamp(utc)).total_seconds() == offset
    utc_t = pd.Timestamp(utc) + pd.to_timedelta([0, 10, 20], unit="s")
    gps_t = pd.Timestamp(gps) + pd.to_timedelta([0, 10, 20], unit="s")
    sc_utc = Spacecraft(state=state, time=utc_t, tscale_input="UTC")
    sc_gps = Spacecraft(state=state, time=gps_t, tscale_input="GPS")
    np.testing.assert_allclose(sc_utc.sspice_gps, sc_gps.sspice_gps, atol=1e-6)
    np.testing.assert_allclose(sc_utc.sspice_et, sc_gps.sspice_et, atol=1e-6)
    # and the GPS axis really is the GPS stamp itself (UTC + offset)
    j2000 = pd.Timestamp("2000-01-01 12:00:00")
    np.testing.assert_allclose(
        sc_utc.sspice_gps, (gps_t - j2000).total_seconds(), atol=1e-6
    )


def test_sc09_scale_offsets():
    """SC-09: TAI and GPS labels of the same stamps differ by 19 s.

    Expected: GPS = TAI - 19 s, so stamps labelled 'TAI' have GPS seconds 19 s
    *smaller* than the same stamps labelled 'GPS' (the instant labelled TAI is
    19 s ahead of the GPS-labelled one, so its GPS reading, TAI - 19 s, is the
    same stamp minus 19 s); for the same reason the ET of the TAI-labelled
    stamps is 19 s smaller, not larger. ET - TAI = 32.184 s (+ periodic term
    < 2 ms), i.e. ET - GPS = 51.184 s on the TAI object's own GPS axis.
    """
    state, time = make_circular_orbit(n=4)
    gps = Spacecraft(state=state, time=time, tscale_input="GPS")
    tai = Spacecraft(state=state, time=time, tscale_input="TAI")
    np.testing.assert_allclose(tai.sspice_gps - gps.sspice_gps, -19.0, atol=1e-6)
    np.testing.assert_allclose(tai.sspice_et - gps.sspice_et, -19.0, atol=1e-6)
    np.testing.assert_allclose(tai.sspice_et - tai.sspice_gps, ET_MINUS_GPS, atol=2e-3)


# ---------------------------------------------------------------------------
# SC-10: spherical
# ---------------------------------------------------------------------------
def test_sc10_spherical():
    """SC-10: ``spherical()`` returns ``[r, lat, lon]`` in rad.

    Expected (geometry): +x -> lat 0, lon 0; +y -> lat 0, lon pi/2; +z -> lat
    pi/2, lon 0; -x -> lon pi; -z -> lat -pi/2; (1,1,1)*R/sqrt(3) ->
    lat = arcsin(1/sqrt(3)), lon = pi/4.
    """
    big = 7.0e6
    d = big / np.sqrt(3.0)
    pos = np.array(
        [
            [big, 0, 0],
            [0, big, 0],
            [0, 0, big],
            [-big, 0, 0],
            [0, 0, -big],
            [d, d, d],
        ]
    )
    state = np.hstack([pos, np.zeros_like(pos)])
    time = pd.date_range("2018-01-02", periods=len(pos), freq="10s")
    sph = Spacecraft(state=state, time=time, tscale_input="GPS").spherical()
    expected = np.array(
        [
            [big, 0.0, 0.0],
            [big, 0.0, np.pi / 2],
            [big, np.pi / 2, 0.0],
            [big, 0.0, np.pi],
            [big, -np.pi / 2, 0.0],
            [big, np.arcsin(1 / np.sqrt(3.0)), np.pi / 4],
        ]
    )
    assert sph.shape == (6, 3)
    np.testing.assert_allclose(sph[:, 0], expected[:, 0], rtol=0, atol=1e-3)  # m
    np.testing.assert_allclose(sph[:, 1:], expected[:, 1:], rtol=0, atol=1e-12)  # rad


# ---------------------------------------------------------------------------
# SC-11, SC-11b: SP3 loader (synthetic files)
# ---------------------------------------------------------------------------
GPS_EPOCH = pd.Timestamp("1980-01-06")


def _sp3_header(epochs) -> str:
    """Header consistent with the body: position+velocity (``#cV``), the first
    epoch, the GPS week / seconds of week, step, MJD and fractional day."""
    t0 = epochs[0]
    step = (epochs[1] - epochs[0]).total_seconds() if len(epochs) > 1 else 0.0
    week, rem = divmod((t0 - GPS_EPOCH).total_seconds(), 7 * 86400.0)
    mjd = (t0.normalize() - pd.Timestamp("1858-11-17")).days
    frac = (t0 - t0.normalize()).total_seconds() / 86400.0
    sec = t0.second + t0.microsecond / 1e6
    return (
        f"#cV{t0.year:4d} {t0.month:2d} {t0.day:2d} {t0.hour:2d} {t0.minute:2d} "
        f"{sec:11.8f} {len(epochs):7d} ORBIT IGb14 HLM  XXX\n"
        f"## {int(week):4d} {rem:15.8f} {step:14.8f} {mjd:5d} {frac:15.13f}\n"
        "%c L  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n"
        "/* synthetic SP3 for contigo tests\n"
    )


def _epoch_line(t: pd.Timestamp) -> str:
    return (
        f"*  {t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} "
        f"{t.second + t.microsecond / 1e6:11.8f}\n"
    )


def _sat_lines(sat: str, pos_km, vel_dms) -> str:
    """``P`` and ``V`` records: F14.6 columns [4:18], [18:32], [32:46]."""
    p = "".join(f"{c:14.6f}" for c in pos_km)
    v = "".join(f"{c:14.6f}" for c in vel_dms)
    return f"P{sat:>3s}{p}{999999.999999:14.6f}\nV{sat:>3s}{v}{999999.999999:14.6f}\n"


def _write_sp3(path, epochs, records) -> None:
    """``records[(epoch_index, sat)] = (pos_km, vel_dm_per_s)``."""
    body = _sp3_header(epochs)
    for i, t in enumerate(epochs):
        body += _epoch_line(t)
        for (j, sat), (pos, vel) in records.items():
            if j == i:
                body += _sat_lines(sat, pos, vel)
    path.write_text(body + "EOF\n")


SP3_T0 = pd.Timestamp("2018-01-02 00:00:00")
SP3_EPOCHS = [SP3_T0 + pd.Timedelta(seconds=900 * k) for k in range(3)]
#: km and dm/s as written, with the SI values (m, m/s) expected after loading
SP3_A = {
    (0, "L48"): ((7000.0, 0.0, 100.5), (0.0, 75000.0, 10.0)),
    (1, "L48"): ((6900.25, 1200.5, 100.5), (-1000.0, 74000.0, 10.0)),
    (2, "L48"): ((6700.0, 2300.125, 100.5), (-2000.0, 73000.0, 10.0)),
}
SP3_B = {
    (0, "L49"): ((0.0, 7100.0, -50.0), (-75000.0, 0.0, 5.0)),
    (1, "L49"): ((1000.0, 7000.0, -50.0), (-74000.0, 1000.0, 5.0)),
    (2, "L49"): ((2000.0, 6900.0, -50.0), (-73000.0, 2000.0, 5.0)),
}


def _si(rec):
    """Expected SI state row from an SP3 record: km -> m, dm/s -> m/s."""
    pos, vel = rec
    return [c * 1000.0 for c in pos] + [c / 10.0 for c in vel]


def test_sc11_sp3_one_satellite(tmp_path):
    """SC-11: 1-satellite SP3 -> km to m, dm/s to m/s, correct epochs.

    Expected: ``(7000, 0, 100.5)`` km becomes ``7.0e6`` m ... and
    ``(0, 75000, 10)`` dm/s becomes ``(0, 7500, 1)`` m/s (1 dm = 0.1 m);
    epochs 2018-01-02 00:00:00, +900 s, +1800 s (GPS time).
    """
    path = tmp_path / "one_sat.sp3"
    _write_sp3(path, SP3_EPOCHS, SP3_A)
    sc = Spacecraft(state_file=path, tscale_input="GPS")
    assert sc.N == 3
    expected = np.array([_si(SP3_A[(k, "L48")]) for k in range(3)])
    np.testing.assert_allclose(sc.state_ecef, expected, rtol=1e-12)
    assert (sc.stime == pd.DatetimeIndex(SP3_EPOCHS)).all()
    np.testing.assert_allclose(np.diff(sc.sspice_gps), 900.0, atol=1e-6)


def test_sc11_sp3_two_satellites(tmp_path):
    """SC-11: 2-satellite SP3 -> every record gets its own epoch.

    Expected: 3 epochs x 2 satellites = 6 rows; each satellite's rows carry the
    epochs of the header lines above them and the km/dm/s -> SI values of its
    own records. Satellite IDs are taken from the ``sat`` column.
    """
    path = tmp_path / "two_sat.sp3"
    _write_sp3(path, SP3_EPOCHS, SP3_A | SP3_B)
    sc = Spacecraft(state_file=path, tscale_input="GPS", sc_id_col="sat")
    assert sc.N == 6
    assert sc.n_unique_ids == 2
    parts = sc.split_by_id()
    for recs, sat in ((SP3_A, "L48"), (SP3_B, "L49")):
        (sub,) = [p for k, p in parts.items() if sat in str(k)]
        expected = np.array([_si(recs[(k, sat)]) for k in range(3)])
        np.testing.assert_allclose(sub.state_ecef, expected, rtol=1e-12)
        assert (sub.stime == pd.DatetimeIndex(SP3_EPOCHS)).all()


def test_sc11b_overlapping_sp3_files_drop_duplicates(tmp_path):
    """SC-11b: same satellite and epoch in two daily files is loaded once.

    Decision (plan, R8): duplicates on (ID, time) are dropped, keeping the
    first. File 1 holds epochs 0, 900, 1800 s; file 2 holds 1800, 2700, 3600 s,
    so 1800 s appears twice (file 2's copy has x shifted by 1 km). Expected:
    5 rows, unique times, and the 1800 s row equals file 1's (sorted path order).
    """
    epochs2 = [SP3_T0 + pd.Timedelta(seconds=1800 + 900 * k) for k in range(3)]
    rec1 = SP3_A
    rec2 = {
        (0, "L48"): ((6701.0, 2300.125, 100.5), (-2000.0, 73000.0, 10.0)),
        (1, "L48"): ((6500.0, 3300.0, 100.5), (-3000.0, 72000.0, 10.0)),
        (2, "L48"): ((6200.0, 4200.0, 100.5), (-4000.0, 71000.0, 10.0)),
    }
    _write_sp3(tmp_path / "day1.sp3", SP3_EPOCHS, rec1)
    _write_sp3(tmp_path / "day2.sp3", epochs2, rec2)
    sc = Spacecraft(
        state_file=str(tmp_path / "day*.sp3"), tscale_input="GPS", sc_id_col="sat"
    )
    assert sc.N == 5
    assert len(np.unique(sc.sspice_gps)) == 5
    offsets = sc.sspice_gps - sc.sspice_gps.min()
    kept = sc.state_ecef[np.isclose(offsets, 1800.0, rtol=0, atol=1e-3)]
    assert len(kept) == 1
    np.testing.assert_allclose(kept[0], _si(rec1[(2, "L48")]), rtol=1e-12)


def test_sc11_df_sp3_path_input_and_columns(tmp_path):
    """SC-11 (extra): ``df_sp3`` accepts a ``Path`` directly.

    Expected: columns ``sat, time, x, y, z, vx, vy, vz``; the vehicle ID is the
    3-character SP3 ID ``'L48'`` (without the ``P`` record letter); values in
    SI. A ``str`` path gives the same frame; unknown keywords raise TypeError.
    """
    path = tmp_path / "one_sat.sp3"
    _write_sp3(path, SP3_EPOCHS, SP3_A)
    df = df_sp3(path)
    assert list(df.columns) == ["sat", "time", "x", "y", "z", "vx", "vy", "vz"]
    assert list(df["sat"]) == ["L48"] * 3
    assert (pd.DatetimeIndex(df["time"]) == pd.DatetimeIndex(SP3_EPOCHS)).all()
    expected = np.array([_si(SP3_A[(k, "L48")]) for k in range(3)])
    np.testing.assert_allclose(df[STATE_COLS].to_numpy(), expected, rtol=1e-12)
    pd.testing.assert_frame_equal(df_sp3(str(path)), df)
    with pytest.raises(TypeError, match="keyword"):
        df_sp3(path, sep=",")


def test_sc11_sp3_without_velocity_raises(tmp_path):
    """SC-11 (extra): an SP3 file with no ``V`` records raises ValueError.

    Expected: Spacecraft needs velocities, so a position-only file is rejected
    instead of loading NaN velocities.
    """
    path = tmp_path / "pos_only.sp3"
    _write_sp3(path, SP3_EPOCHS, SP3_A)
    text = path.read_text().splitlines(keepends=True)
    path.write_text("".join(ln for ln in text if not ln.startswith("V")))
    with pytest.raises(ValueError, match="velocity"):
        df_sp3(path)
    with pytest.raises(ValueError, match="velocity"):
        Spacecraft(state_file=path, tscale_input="GPS")


def test_sc11_sp3_ignores_correlation_records(tmp_path):
    """SC-11 (extra): ``EP``/``EV`` correlation records are ignored.

    Expected: inserting an ``EP`` and ``EV`` line after every record leaves the
    loaded frame identical to the file without them.
    """
    plain = tmp_path / "plain.sp3"
    _write_sp3(plain, SP3_EPOCHS, SP3_A)
    lines = plain.read_text().splitlines(keepends=True)
    corr = []
    for ln in lines:
        corr.append(ln)
        if ln.startswith("P"):
            corr.append("EP  55   55   55  222 1234567 -1234567 5999999\n")
        elif ln.startswith("V"):
            corr.append("EV  22   22   22  111 1234567  1234567 1234567\n")
    with_corr = tmp_path / "corr.sp3"
    with_corr.write_text("".join(corr))
    pd.testing.assert_frame_equal(df_sp3(with_corr), df_sp3(plain))


def test_sc11_sp3_bad_value_record_dropped(tmp_path, caplog):
    """SC-11 (extra): a record with a 999999.999999 position is dropped.

    Expected: the middle epoch of L48 is written with a bad x position, so
    only epochs 0 and 1800 s remain, and a warning is logged.
    """
    recs = dict(SP3_A)
    pos, vel = recs[(1, "L48")]
    recs[(1, "L48")] = ((999999.999999, pos[1], pos[2]), vel)
    path = tmp_path / "bad.sp3"
    _write_sp3(path, SP3_EPOCHS, recs)
    with caplog.at_level("WARNING"):
        df = df_sp3(path)
    assert len(df) == 2
    assert (pd.DatetimeIndex(df["time"]) == pd.DatetimeIndex(SP3_EPOCHS[::2])).all()
    assert "bad" in caplog.text


def test_sc11b_overlapping_csv_files_drop_duplicates(tmp_path, caplog):
    """SC-11b (extra): the (ID, time) dedup also applies to CSV files.

    Expected: two CSV files of one spacecraft (``sc_id_input``) share 5 of 15
    samples (the second file's copies have x shifted by 1 m); 25 unique rows
    load, the shared rows carry the first file's values, and a warning names
    the count and an example (ID, time) pair.
    """
    df = _frame(n=25)
    first = df.iloc[:15]
    second = df.iloc[10:].copy()
    second["x"] += 1.0
    first.to_csv(tmp_path / "a.csv", index=False)
    second.to_csv(tmp_path / "b.csv", index=False)
    with caplog.at_level("WARNING"):
        sc = Spacecraft(
            state_file=str(tmp_path / "*.csv"),
            tscale_input="GPS",
            sc_id_input="SAT-A",
        )
    assert sc.N == 25
    np.testing.assert_allclose(sc.state_ecef[:15], first[STATE_COLS].to_numpy())
    np.testing.assert_allclose(
        sc.state_ecef[15:], second[STATE_COLS].to_numpy()[5:], rtol=1e-12
    )
    assert "Dropped 5 duplicate" in caplog.text
    assert f"(SAT-A, {pd.Timestamp(df['time'].iloc[10])})" in caplog.text


def test_sc11b_csv_without_ids_keeps_repeated_times(tmp_path, caplog):
    """SC-11b (extra): without IDs, repeated times are kept with a warning.

    Expected: two different spacecraft in two CSVs on the same time grid and
    no ``sc_id_col``/``sc_id_input`` -> all 20 rows load (nothing is dropped,
    as the rows are probably different spacecraft) and a warning suggests
    setting ``sc_id_col``.
    """
    dfa, dfb = _frame(10), _frame(10, alt=500e3, inc_deg=87.0)
    dfa.to_csv(tmp_path / "a.csv", index=False)
    dfb.to_csv(tmp_path / "b.csv", index=False)
    with caplog.at_level("WARNING"):
        sc = Spacecraft(state_file=str(tmp_path / "*.csv"), tscale_input="GPS")
    assert sc.N == 20
    np.testing.assert_allclose(sc.state_ecef[:10], dfa[STATE_COLS], rtol=1e-12)
    np.testing.assert_allclose(sc.state_ecef[10:], dfb[STATE_COLS], rtol=1e-12)
    assert "sc_id_col" in caplog.text
    assert "none were dropped" in caplog.text


def test_sc11_sp3_ids_from_file_by_default(tmp_path):
    """SC-11 (extra): without ``sc_id_col`` SP3 vehicle IDs are the IDs.

    Expected: a 2-satellite SP3 loaded with no ID options keeps all 6 rows
    (no satellite is mistaken for a duplicate) with IDs ``L48`` and ``L49``.
    """
    path = tmp_path / "two_sat.sp3"
    _write_sp3(path, SP3_EPOCHS, SP3_A | SP3_B)
    sc = Spacecraft(state_file=path, tscale_input="GPS")
    assert sc.N == 6
    assert sorted(map(str, sc.unique_ids)) == ["L48", "L49"]
    parts = sc.split_by_id()
    for recs, sat in ((SP3_A, "L48"), (SP3_B, "L49")):
        expected = np.array([_si(recs[(k, sat)]) for k in range(3)])
        np.testing.assert_allclose(parts[sat].state_ecef, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "name, opener",
    [("orbit.sp3.gz", gzip.open), ("orbit.sp3.bz2", bz2.open)],
)
def test_sc11_compressed_sp3_through_spacecraft(tmp_path, name, opener):
    """SC-11 (extra): compressed SP3 files load through Spacecraft.

    Expected: the loader is inferred from the suffix chain (``.sp3.gz`` ->
    sp3) and the compression from the magic bytes; the state equals the
    uncompressed file's.
    """
    plain = tmp_path / "plain.sp3"
    _write_sp3(plain, SP3_EPOCHS, SP3_A)
    packed = tmp_path / name
    with opener(packed, "wb") as f:
        f.write(plain.read_bytes())
    sc = Spacecraft(state_file=packed, tscale_input="GPS")
    expected = np.array([_si(SP3_A[(k, "L48")]) for k in range(3)])
    np.testing.assert_allclose(sc.state_ecef, expected, rtol=1e-12)
    assert list(sc.unique_ids) == ["L48"]


def test_sc11_infer_loader_from_suffix_chain():
    """SC-11 (extra): loader inference uses the full suffix chain.

    Expected: compression suffixes are skipped (``.sp3.Z`` -> sp3,
    ``.csv.gz`` -> csv); a bare ``.gz`` is CSV as before; unknown -> error.
    """
    infer = Spacecraft._infer_loader
    for name in ("a.sp3", "a.sp3.gz", "a.sp3.Z", "a.SP3.zip", "CH.L48.sp3.gz"):
        assert infer(Path(name)) == "sp3", name
    for name in ("a.csv", "a.csv.gz", "a.txt.bz2", "a.gz", "a.2018.gz"):
        assert infer(Path(name)) == "csv", name
    assert infer(Path("a.h5")) == "hdf"
    assert infer(Path("a.dat"), "sp3") == "sp3"
    with pytest.raises(ValueError, match="Unsupported loader"):
        infer(Path("a.dat"))


def test_sc11_sp3_repeated_record_keeps_first_good(tmp_path, caplog):
    """SC-11 (extra): a repeated (satellite, epoch) record keeps the first good.

    Expected: epoch 0 is followed by a second, different L48 P/V pair -> the
    first is kept. At epoch 1 a bad P record precedes a good one -> the good
    one is kept (the bad flag belongs to the record, not the key). At epoch 2
    a bad repeat after a good record does not remove the good one.
    """
    good = {k: _sat_lines("L48", *SP3_A[(k, "L48")]) for k in range(3)}
    other = _sat_lines("L48", (1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    bad_p = _sat_lines("L48", (999999.999999, 0.0, 0.0), (4.0, 5.0, 6.0))
    absent_p = _sat_lines("L48", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    body = _sp3_header(SP3_EPOCHS)
    body += _epoch_line(SP3_EPOCHS[0]) + good[0] + other
    body += _epoch_line(SP3_EPOCHS[1]) + bad_p + good[1]
    body += _epoch_line(SP3_EPOCHS[2]) + good[2] + absent_p
    path = tmp_path / "rep.sp3"
    path.write_text(body + "EOF\n")
    with caplog.at_level("WARNING"):
        df = df_sp3(path)
    expected = np.array([_si(SP3_A[(k, "L48")]) for k in range(3)])
    np.testing.assert_allclose(df[STATE_COLS].to_numpy(), expected, rtol=1e-12)
    assert (pd.DatetimeIndex(df["time"]) == pd.DatetimeIndex(SP3_EPOCHS)).all()
    assert "repeated" in caplog.text


def test_sc11_sp3_blank_or_short_fields_dropped(tmp_path, caplog):
    """SC-11 (extra): blank/short numeric fields drop the record with a warning.

    Expected: the epoch-1 P record is truncated after x, so its y and z are
    missing; that record (and so its epoch) is dropped and the warning names
    the file and the line number.
    """
    path = tmp_path / "short.sp3"
    _write_sp3(path, SP3_EPOCHS, SP3_A)
    lines = path.read_text().splitlines(keepends=True)
    p_idx = [i for i, ln in enumerate(lines) if ln.startswith("P")][1]
    lines[p_idx] = lines[p_idx][:18] + "\n"
    path.write_text("".join(lines))
    with caplog.at_level("WARNING"):
        df = df_sp3(path)
    assert (pd.DatetimeIndex(df["time"]) == pd.DatetimeIndex(SP3_EPOCHS[::2])).all()
    assert "short.sp3" in caplog.text
    assert f"line(s) {p_idx + 1}" in caplog.text


def test_sc11_sp3_non_utf8_comment(tmp_path):
    """SC-11 (extra): a non-UTF-8 byte in a comment line does not break loading.

    Expected: a Latin-1 comment (``\\xe9``) is decoded with replacement and the
    frame equals the clean file's.
    """
    plain = tmp_path / "plain.sp3"
    _write_sp3(plain, SP3_EPOCHS, SP3_A)
    raw = plain.read_bytes().replace(b"/* synthetic", b"/* caf\xe9 synthetic", 1)
    odd = tmp_path / "latin1.sp3"
    odd.write_bytes(raw)
    pd.testing.assert_frame_equal(df_sp3(odd), df_sp3(plain))


def test_sc11_sp3_rejects_unit_input_km(tmp_path):
    """SC-11 (extra): SP3 is converted to metres, so ``unit_input='km'`` fails.

    Expected: ValueError explaining that unit_input should stay 'm'.
    """
    path = tmp_path / "one_sat.sp3"
    _write_sp3(path, SP3_EPOCHS, SP3_A)
    with pytest.raises(ValueError, match="unit_input"):
        Spacecraft(state_file=path, tscale_input="GPS", unit_input="km")


def test_sc11_sp3_unix_compress_rejected(tmp_path):
    """SC-11 (extra): a unix-compress (.Z) file raises a clear error.

    Expected: the LZW magic bytes ``1f 9d`` are recognised and rejected.
    """
    path = tmp_path / "orbit.sp3.Z"
    path.write_bytes(b"\x1f\x9d\x90" + b"\x00" * 20)
    with pytest.raises(ValueError, match="unix-compress"):
        df_sp3(path)


# ---------------------------------------------------------------------------
# SC-12: state_data cache
# ---------------------------------------------------------------------------
def test_sc12_state_data_cache(sc_single):
    """SC-12: ``state_data`` is cached and invalidated by ``load_from_arrays``.

    Expected: two accesses return the same object; after reloading 10 samples
    the next access is a different object holding the new (10, 6) state.
    """
    sd = sc_single.state_data
    assert sc_single.state_data is sd
    assert sd.N == 600
    np.testing.assert_array_equal(sd.r, sc_single.state_ecef[:, :3])
    np.testing.assert_array_equal(sd.v, sc_single.state_ecef[:, 3:])

    state, time = make_circular_orbit(n=10, alt=600e3)
    sc_single.load_from_arrays(state, time, np.full(10, "SAT-Z"))
    sd2 = sc_single.state_data
    assert sd2 is not sd
    assert sd2.N == 10
    np.testing.assert_array_equal(sd2.state_ecef, state)
    assert list(sd2.unique_ids) == ["SAT-Z"]


def test_sc12_reload_refreshes_time_axes(sc_single):
    """SC-12 (extra): after ``load_from_arrays`` the time axes match the new N.

    Expected: a reload with 10 samples gives 10 GPS seconds, 10 ET values and
    10 UTC datetimes (as a freshly built Spacecraft would).
    """
    state, time = make_circular_orbit(n=10, alt=600e3)
    sc_single.load_from_arrays(state, time, np.full(10, "SAT-Z"))
    assert len(sc_single.sspice_gps) == 10
    assert len(sc_single.sspice_et) == 10
    assert len(sc_single.sc_utc) == 10
    # the cached grouped view must carry the refreshed axes too
    sd = sc_single.state_data
    assert len(sd.sspice_gps) == sd.N == 10


def test_sc12_reload_from_file_refreshes_everything(tmp_path):
    """SC-12 (extra): ``load_from_file`` on an existing object is fully normalised.

    Expected: after reloading a 7-row km CSV into a 20-row km spacecraft, N,
    the time axes, the per-row properties (from the ``cd`` column) and the
    SI state all equal those of a Spacecraft freshly built from that CSV,
    with the km conversion applied once.
    """
    df1 = _frame(n=20)
    df1[STATE_COLS] /= 1000.0
    df1["cd"] = 2.0
    df1.to_csv(tmp_path / "first.csv", index=False)

    df2 = _frame(n=7, alt=600e3, t0="2024-03-01T00:00:00")
    state2 = df2[STATE_COLS].to_numpy()
    df2[STATE_COLS] /= 1000.0
    df2["cd"] = np.linspace(2.1, 2.7, 7)
    df2.to_csv(tmp_path / "second.csv", index=False)

    kw = dict(tscale_input="GPS", unit_input="km", cd_col="cd")
    sc = Spacecraft(state_file=tmp_path / "first.csv", **kw)
    assert sc.N == 20
    sd_old = sc.state_data

    sc.load_from_file(tmp_path / "second.csv")
    fresh = Spacecraft(state_file=tmp_path / "second.csv", **kw)

    assert sc.N == 7
    for name in ("sspice_et", "sspice_gps"):
        np.testing.assert_array_equal(getattr(sc, name), getattr(fresh, name))
    assert list(sc.sc_utc) == list(fresh.sc_utc)
    np.testing.assert_allclose(sc.cd_arr, df2["cd"].to_numpy(), rtol=1e-15)
    np.testing.assert_allclose(sc.state_ecef, state2, rtol=1e-12)
    np.testing.assert_array_equal(sc.state_ecef, fresh.state_ecef)
    sd = sc.state_data
    assert sd is not sd_old
    assert sd.N == len(sd.sspice_gps) == len(sd.cd) == 7
