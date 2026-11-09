"""Tests for ``contigo.contigo_utils.time_utils`` (TU-01 .. TU-05, PR T2).

Expected values come from the SPICE leap-second kernel ``naif0012.tls`` and
from the definitions of the time scales:

- ``ET`` (TDB) = ``TAI`` + 32.184 s + a periodic term below 2 ms.
- ``TAI`` = ``GPS`` + 19 s, always.
- ``GPS`` - ``UTC`` = 13 s in Aug 2005, 17 s in Jun 2016, 18 s in Jun 2017.

``spice_time`` takes a ``DatetimeIndex`` and returns seconds past J2000 on the
output scale (``UTC`` input is converted through ``spice.datetime2et``).

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import os
import shutil

import pandas as pd
import pytest
import spiceypy as spice

import contigo.config as config
from contigo.contigo_utils import utils
from contigo.contigo_utils.time_utils import check_lpsk, spice_time

LEAP_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
J2000 = pd.DatetimeIndex(["2000-01-01 12:00:00"])


def _one(stamp: str, tscale: str, out_scale: str) -> float:
    """``spice_time`` of a single timestamp as a float."""
    return float(spice_time(pd.DatetimeIndex([stamp]), tscale, out_scale)[0])


# ---------------------------------------------------------------------------
# TU-01 .. TU-02: conversions
# ---------------------------------------------------------------------------
def test_tu01_gps_j2000_to_et():
    """TU-01: GPS 2000-01-01 12:00:00 is ET 51.184 s (32.184 + 19) within 2 ms.

    Source: ET - TAI = 32.184 s + periodic term (< 2 ms); TAI - GPS = 19 s.
    """
    out = spice_time(J2000, "GPS", "ET")
    assert out.shape == (1,)
    assert out[0] == pytest.approx(51.184, abs=2e-3)


@pytest.mark.parametrize(
    "stamp, gps_minus_utc",
    [
        ("2005-08-16 00:00:00", 13.0),
        ("2016-06-01 00:00:00", 17.0),
        ("2017-06-01 00:00:00", 18.0),
    ],
)
def test_tu02_utc_to_gps_offsets(stamp, gps_minus_utc):
    """TU-02: leap seconds between UTC and GPS (``naif0012.tls``).

    Sign: a stamp T read as UTC is the instant whose GPS reading is
    ``T + (GPS - UTC)`` because GPS runs ahead of UTC. So the GPS seconds of
    the UTC-labelled stamp minus the GPS seconds of the same stamp labelled
    GPS equal ``+(GPS - UTC)`` = +13 / +17 / +18 s. (The reverse direction,
    GPS-labelled minus UTC-labelled, is the negative of this.)
    """
    from_utc = _one(stamp, "UTC", "GPS")
    from_gps = _one(stamp, "GPS", "GPS")
    assert from_utc - from_gps == pytest.approx(gps_minus_utc, abs=1e-6)


def test_tu02_tai_gps_offset_is_19_seconds():
    """TU-02: TAI - GPS = 19 s, the constant that TU-01 relies on (definition)."""
    assert _one("2010-03-04 05:06:07", "GPS", "TAI") - _one(
        "2010-03-04 05:06:07", "TAI", "TAI"
    ) == pytest.approx(19.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TU-03 .. TU-04: validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tscale, out_scale", [("XYZ", "ET"), ("GPS", "XYZ")])
def test_tu03_invalid_scale_raises(tscale, out_scale):
    """TU-03: an unknown tscale or out_scale raises ``ValueError`` (source: code)."""
    with pytest.raises(ValueError, match="must be one of"):
        spice_time(J2000, tscale, out_scale)


def test_tu03_lowercase_tscale_is_accepted():
    """TU-03: lower-case input ``tscale`` is upper-cased, giving the TU-01 value."""
    out = spice_time(J2000, "gps", "ET")
    assert out[0] == pytest.approx(51.184, abs=2e-3)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="F15/R10: out_scale is not upper-cased",
)
def test_tu03_lowercase_out_scale_is_accepted():
    """TU-03: lower-case ``out_scale`` is accepted (DECISION 3, R10).

    Expected value as TU-01. Currently only ``tscale`` is upper-cased, so
    ``'et'`` raises ``ValueError``.
    """
    out = spice_time(J2000, "GPS", "et")
    assert out[0] == pytest.approx(51.184, abs=2e-3)


@pytest.mark.parametrize("tscale", ["GPS", "TAI", "ET", "TDB"])
def test_tu04_utc_output_not_implemented(tscale):
    """TU-04: converting a non-UTC input to UTC raises ``ValueError`` (code)."""
    with pytest.raises(ValueError, match="not yet impl"):
        spice_time(J2000, tscale, "UTC")


# ---------------------------------------------------------------------------
# TU-05: check_lpsk
# ---------------------------------------------------------------------------
def test_tu05_check_lpsk_is_idempotent(monkeypatch):
    """TU-05: with the kernel loaded, a second call neither furnshes nor loads.

    Source: ``ktotal('ALL')`` before and after must be equal and ``furnsh``
    must not be called (the ``spice_offline`` fixture loaded the kernel).
    """
    check_lpsk()  # make sure the leap-second kernel is loaded
    before = spice.ktotal("ALL")
    calls: list[str] = []
    monkeypatch.setattr(spice, "furnsh", lambda path: calls.append(path))
    check_lpsk()
    check_lpsk()
    assert calls == []
    assert spice.ktotal("ALL") == before


def _pretend_nothing_loaded(monkeypatch) -> list[str]:
    """Make ``check_lpsk`` see no loaded kernels and record ``furnsh`` calls.

    The real kernel pool is left untouched, so other tests are not affected.
    """
    calls: list[str] = []
    monkeypatch.setattr(spice, "ktotal", lambda kind: 0)
    monkeypatch.setattr(spice, "furnsh", lambda path: calls.append(path))
    return calls


def test_tu05_check_lpsk_furnishes_existing_file(tmp_path, monkeypatch):
    """TU-05: kernel present in DATA_DIR but not loaded is furnished, no download.

    Source: branch ``os.path.exists(lp_kernel) and leaps_f not in sp_loaded``.
    """
    target = tmp_path / config.LEAP_FILE
    shutil.copy(os.path.join(config.DATA_DIR, config.LEAP_FILE), target)
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    downloads: list[tuple[str, str]] = []
    monkeypatch.setattr(utils, "dl_file", lambda url, fn: downloads.append((url, fn)))
    calls = _pretend_nothing_loaded(monkeypatch)

    check_lpsk()

    assert calls == [str(target)]
    assert downloads == []


def test_tu05_check_lpsk_downloads_missing_file(tmp_path, monkeypatch):
    """TU-05: a missing kernel is downloaded from NAIF, then furnished.

    The fake ``dl_file`` copies the bundled ``naif0012.tls`` into place and
    records the URL. Expected URL: the NAIF generic_kernels/lsk location.
    """
    bundled = os.path.join(config.DATA_DIR, config.LEAP_FILE)
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    downloads: list[tuple[str, str]] = []

    def fake_dl(url, filename):
        downloads.append((url, filename))
        shutil.copy(bundled, filename)

    monkeypatch.setattr(utils, "dl_file", fake_dl)
    calls = _pretend_nothing_loaded(monkeypatch)
    expected = os.path.join(str(tmp_path), config.LEAP_FILE)
    assert not os.path.exists(expected)

    check_lpsk()

    assert downloads == [(LEAP_URL, expected)]
    assert calls == [expected]
    assert os.path.exists(expected)
