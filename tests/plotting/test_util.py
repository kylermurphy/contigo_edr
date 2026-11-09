"""Tests for contigo.plotting._util (PL-01 to PL-08).

added: 09/11/2026 Claude Code
"""

import numpy as np
import pytest

from contigo.constellation import Constellation
from contigo.plotting._util import as_spacecraft_dict, decimate, split_antimeridian
from contigo.spacecraft import Spacecraft


def test_pl01_spacecraft_is_split_by_id(multi_sc: Spacecraft) -> None:
    """PL-01: a multi-ID Spacecraft becomes one Spacecraft per ID."""
    out = as_spacecraft_dict(multi_sc)
    assert [str(k) for k in out] == ["A", "B", "C"]
    assert all(isinstance(v, Spacecraft) for v in out.values())
    assert all(len(v.sspice_gps) == 10 for v in out.values())


def test_pl02_constellation_uses_its_dict(constellation: Constellation) -> None:
    """PL-02: a Constellation gives its own spacecraft dict (original keys)."""
    out = as_spacecraft_dict(constellation)
    assert list(out) == list(constellation.spacecraft)
    for k in out:
        assert out[k] is constellation.spacecraft[k]


def test_pl03_ids_filter_is_string_safe(constellation: Constellation) -> None:
    """PL-03: ids filter by str(); original keys and order are kept."""
    out = as_spacecraft_dict(constellation, ids=["C", "A"])
    assert [str(k) for k in out] == ["A", "C"]
    assert list(out) == [k for k in constellation.spacecraft if str(k) in ("A", "C")]
    assert [str(k) for k in as_spacecraft_dict(constellation, ids="B")] == ["B"]
    assert [str(k) for k in as_spacecraft_dict(constellation, ids=np.array(["B"]))] == [
        "B"
    ]


def test_pl04_bad_inputs(constellation: Constellation) -> None:
    """PL-04: unknown ID -> ValueError naming IDs; other types -> TypeError."""
    with pytest.raises(ValueError) as exc:
        as_spacecraft_dict(constellation, ids=["A", "Z"])
    msg = str(exc.value)
    assert "Z" in msg
    for sid in ("A", "B", "C"):
        assert sid in msg
    with pytest.raises(TypeError):
        as_spacecraft_dict({"A": 1})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        as_spacecraft_dict(None)  # type: ignore[arg-type]


def test_pl05_decimate_small_n_returns_arange() -> None:
    """PL-05: n <= max_points returns arange(n)."""
    np.testing.assert_array_equal(decimate(5, 10), np.arange(5))
    np.testing.assert_array_equal(decimate(10, 10), np.arange(10))
    assert decimate(0, 10).size == 0


def test_pl06_decimate_keeps_ends_and_is_bounded() -> None:
    """PL-06: large n gives sorted unique indices with 0 and n-1."""
    idx = decimate(100_000, 2000)
    assert idx[0] == 0
    assert idx[-1] == 99_999
    assert 2 <= idx.size <= 2000
    assert np.all(np.diff(idx) > 0)
    np.testing.assert_array_equal(decimate(11, 2), [0, 10])
    idx = decimate(11, 3)
    assert list(idx) == [0, 5, 10]


def test_pl07_decimate_validation() -> None:
    """PL-07: max_points < 2 or n < 0 raises ValueError."""
    with pytest.raises(ValueError):
        decimate(10, 1)
    with pytest.raises(ValueError):
        decimate(10, 0)
    with pytest.raises(ValueError):
        decimate(-1, 10)


def test_pl08_split_antimeridian() -> None:
    """PL-08: NaN inserted at wraps in lon and companions; length checked."""
    lon = np.array([3.0, 3.1, -3.1, -3.0, -2.9, 3.0, 3.1])
    lat = np.arange(7.0)
    lon_o, lat_o = split_antimeridian(lon, lat)
    assert lon_o.size == lat_o.size == 9
    np.testing.assert_array_equal(np.isnan(lon_o), np.isnan(lat_o))
    assert list(np.flatnonzero(np.isnan(lon_o))) == [2, 6]
    np.testing.assert_array_equal(lon_o[~np.isnan(lon_o)], lon)
    np.testing.assert_array_equal(lat_o[~np.isnan(lat_o)], lat)

    # no wrap: unchanged; lon only: 1-tuple
    (same,) = split_antimeridian(np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(same, [0.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        split_antimeridian(lon, lat[:-1])
