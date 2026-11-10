"""Tests for ``contigo.constellation.Constellation`` (PR T1, catalogue §4.2).

Rows CO-01 to CO-04 of ``docs/testing.md``. Every test docstring starts with
its catalogue row ID and states where the expected numbers come from.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _support import MULTI_MEMBERS, make_circular_orbit

from contigo.constellation import Constellation
from contigo.spacecraft import Spacecraft

ET_MINUS_GPS = 51.184


def _member(sc_id: str, offset_s: float, n: int = 5, **orbit) -> Spacecraft:
    """Single-ID GPS-time Spacecraft sampled every 10 s from ``offset_s``."""
    state, time = make_circular_orbit(n=n, **orbit)
    time = time + pd.Timedelta(seconds=offset_s)
    return Spacecraft(
        state=state,
        time=time,
        sc_id_input=np.full(n, sc_id),
        tscale_input="GPS",
        cd=2.5,
    )


def _assert_union_axes(con: Constellation, n_expected: int) -> None:
    """The three union axes: same length, strictly increasing, ET - GPS aligned."""
    for arr in (con.sspice_gps, con.sspice_et, con.sc_utc):
        assert len(arr) == n_expected
    assert (np.diff(con.sspice_gps) > 0).all()
    assert (np.diff(con.sspice_et) > 0).all()
    assert np.abs(con.sspice_et - con.sspice_gps - ET_MINUS_GPS).max() < 2e-3
    utc = [pd.Timestamp(t) for t in con.sc_utc]
    assert utc == sorted(utc)
    assert len(set(utc)) == n_expected


def test_co01_from_multi_id_spacecraft(sc_multi):
    """CO-01: Constellation from a multi-ID ``Spacecraft``.

    Expected: three IDs (``MULTI_MEMBERS``); all members share the same 200
    epochs, so the union axis has 200 sorted, unique entries.
    """
    con = Constellation(spacecraft=sc_multi)
    assert len(con) == 3
    assert con.ids == sorted(m[0] for m in MULTI_MEMBERS)
    _assert_union_axes(con, 200)
    np.testing.assert_allclose(np.diff(con.sspice_gps), 10.0, atol=1e-6)
    assert sum(sc.N for sc in con) == 600


def test_co01_from_kwargs(sc_multi):
    """CO-01: Constellation from ``Spacecraft`` kwargs (arrays).

    Expected: the same arrays as ``sc_multi`` passed as keywords give the same
    IDs, per-member states and union axes as building the Spacecraft first.
    """
    con_ref = Constellation(spacecraft=sc_multi)
    con = Constellation(
        state=sc_multi.state_ecef,
        time=sc_multi.stime,
        sc_id_input=sc_multi.sc_id,
        tscale_input="GPS",
        cd=sc_multi.cd_arr,
    )
    assert con.ids == con_ref.ids
    np.testing.assert_array_equal(con.sspice_gps, con_ref.sspice_gps)
    for sc_id in con.ids:
        np.testing.assert_array_equal(con[sc_id].state_ecef, con_ref[sc_id].state_ecef)
        np.testing.assert_array_equal(con[sc_id].cd_arr, con_ref[sc_id].cd_arr)


def test_co01_union_of_offset_members():
    """CO-01: the union axis merges different, partly overlapping epochs.

    Expected: A has epochs 0..40 s, B has 20..60 s (10 s spacing): the union is
    0, 10, ..., 60 = 7 entries, with the shared epochs 20, 30, 40 counted once.
    """
    a, b = _member("A", 0.0), _member("B", 20.0)
    state = np.vstack([a.state_ecef, b.state_ecef])
    time = np.concatenate([a.stime.values, b.stime.values])
    ids = np.concatenate([a.sc_id, b.sc_id])
    con = Constellation(state=state, time=time, sc_id_input=ids, tscale_input="GPS")
    assert con.ids == ["A", "B"]
    _assert_union_axes(con, 7)
    np.testing.assert_allclose(
        con.sspice_gps - con.sspice_gps[0], np.arange(7) * 10.0, atol=1e-6
    )
    # members keep their own epochs
    assert con["A"].N == con["B"].N == 5


def test_co02_from_list_of_spacecraft():
    """CO-02: a list of single-ID Spacecraft is merged.

    Expected: ids [A, B, C]; epochs A 0..40 s, B 20..60 s, C 100..140 s give
    7 + 5 = 12 union epochs.
    """
    con = Constellation(
        spacecraft=[_member("A", 0.0), _member("B", 20.0), _member("C", 100.0)]
    )
    assert con.ids == ["A", "B", "C"]
    assert len(con) == 3
    _assert_union_axes(con, 12)
    assert all(sc.N == 5 for sc in con)

    # a tuple is accepted too
    con_t = Constellation(spacecraft=(_member("A", 0.0), _member("B", 20.0)))
    assert con_t.ids == ["A", "B"]


def test_co02_list_errors():
    """CO-02: duplicate IDs and empty list -> ``ValueError``; non-Spacecraft
    -> ``TypeError``.

    Expected: per the catalogue; the same checks apply to a non-Spacecraft
    scalar passed as ``spacecraft``.
    """
    with pytest.raises(ValueError, match="Duplicate"):
        Constellation(spacecraft=[_member("A", 0.0), _member("A", 100.0)])
    with pytest.raises(ValueError, match="empty"):
        Constellation(spacecraft=[])
    with pytest.raises(TypeError):
        Constellation(spacecraft=[_member("A", 0.0), "not a spacecraft"])
    with pytest.raises(TypeError):
        Constellation(spacecraft="not a spacecraft")


def test_co03_add_does_not_modify_operands():
    """CO-03: ``+`` builds a new Constellation and leaves both operands alone.

    Expected: A (0..40 s) + B (20..60 s): result has ids [A, B] and 7 aligned
    union epochs (ET - GPS = 51.184 s within 2 ms); A still has one member and 5
    epochs, B likewise.
    """
    ca = Constellation(spacecraft=_member("A", 0.0))
    cb = Constellation(spacecraft=_member("B", 20.0))
    gps_a, gps_b = ca.sspice_gps.copy(), cb.sspice_gps.copy()

    total = ca + cb
    assert total is not ca and total is not cb
    assert total.ids == ["A", "B"]
    _assert_union_axes(total, 7)
    assert total["A"] is ca["A"] and total["B"] is cb["B"]

    assert ca.ids == ["A"] and cb.ids == ["B"]
    np.testing.assert_array_equal(ca.sspice_gps, gps_a)
    np.testing.assert_array_equal(cb.sspice_gps, gps_b)
    _assert_union_axes(ca, 5)
    _assert_union_axes(cb, 5)


def test_co03_iadd_updates_in_place():
    """CO-03: ``+=`` mutates and returns the left operand.

    Expected: after ``ca += cb`` the object is the same, has ids [A, B] and the
    7 aligned union epochs; ``cb`` is unchanged.
    """
    ca = Constellation(spacecraft=_member("A", 0.0))
    cb = Constellation(spacecraft=_member("B", 20.0))
    before = ca
    ca += cb
    assert ca is before
    assert ca.ids == ["A", "B"]
    _assert_union_axes(ca, 7)
    assert cb.ids == ["B"]
    _assert_union_axes(cb, 5)


def test_co03_add_errors():
    """CO-03: duplicate IDs raise ``ValueError``; foreign operands are refused.

    Expected: ``A + A`` -> ValueError for ``+`` and ``+=``; adding an int
    -> TypeError (``NotImplemented`` from ``__add__``).
    """
    ca = Constellation(spacecraft=_member("A", 0.0))
    other_a = Constellation(spacecraft=_member("A", 100.0))
    with pytest.raises(ValueError, match="Duplicate"):
        ca + other_a
    with pytest.raises(ValueError, match="Duplicate"):
        ca += other_a
    assert ca.ids == ["A"]
    with pytest.raises(TypeError):
        ca + 1
    with pytest.raises(TypeError):
        ca += 1


def test_co04_container_protocol(constellation_fixture):
    """CO-04: ``__iter__``, ``__getitem__``, ``__len__`` and ``repr``.

    Expected: three members (``sc_multi``); iteration yields the member
    ``Spacecraft`` objects in ID order; ``repr`` names the count and IDs.
    """
    con = constellation_fixture
    ids = sorted(m[0] for m in MULTI_MEMBERS)
    assert len(con) == 3
    assert con.ids == ids
    members = list(con)
    assert len(members) == 3
    assert all(isinstance(sc, Spacecraft) for sc in members)
    for sc_id, sc in zip(ids, members, strict=True):
        assert con[sc_id] is sc
        assert list(sc.unique_ids) == [sc_id]
    with pytest.raises(KeyError):
        con["NOT-A-SAT"]
    assert repr(con) == f"Constellation(n_spacecraft=3, ids={ids})"
