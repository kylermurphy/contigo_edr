"""Tests for ``SolarSystemEnvironment``, the cached ephemeris container (PR T3).

Rows EP-01b, EP-04 and EP-05 of docs/testing.md. The cache is
exercised with ``FakeProvider`` (counts calls, records requested ET); EP-01b
uses the real ``SPICEEphem`` with the bundled kernels.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import numpy as np
import pytest
import spiceypy as spice
from _support import FakeProvider

from contigo.contigo_utils.constants import GMc
from contigo.ephemeris.spice_ephem import SPICEEphem
from contigo.solar_system_ephem import SolarSystemEnvironment

BODIES = ["SUN", "MOON"]
ET_OFFSET = 51.184  # ET - GPS at J2000, only used to build plausible inputs


def _times(gps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ephem_time, gps_time, utc_time) arrays for the given GPS seconds."""
    gps = np.atleast_1d(np.asarray(gps, dtype=float))
    utc = np.datetime64("2018-01-02T00:00:00") + (gps * 1e3).astype("timedelta64[ms]")
    return gps + ET_OFFSET, gps, utc


def _get(env: SolarSystemEnvironment, gps) -> np.ndarray:
    """Positions (B,N,3) returned by ``env.get_ephem`` for the GPS seconds."""
    et, g, utc = _times(gps)
    out = env.get_ephem(ephem_time=et, gps_time=g, utc_time=utc)
    assert len(out) == 4
    return out[3]


def _env(provider, tolerance=0.001, **kw) -> SolarSystemEnvironment:
    return SolarSystemEnvironment(BODIES, tolerance, provider, **kw)


# ---------------------------------------------------------------------------
# EP-01b
# ---------------------------------------------------------------------------
def test_ep01b_env_bit_identical_to_direct_spkpos(esa_constellation):
    """EP-01b: positions via the environment equal a direct ``spkpos`` x 1000.

    Expected values come from ``spice.spkpos(body, et, 'ITRF93', 'NONE',
    'EARTH')[0] * 1000`` at the ESA epochs (integer GPS seconds), compared with
    ``np.array_equal`` (bit-identical).
    """
    sc = esa_constellation.spacecraft["ESA"]
    assert np.array_equal(sc.sspice_gps, np.round(sc.sspice_gps))
    env = SolarSystemEnvironment(BODIES, 0.001, SPICEEphem())

    et, gps, utc, r = env.get_ephem(
        ephem_time=sc.sspice_et, gps_time=sc.sspice_gps, utc_time=sc.sc_utc
    )

    assert r.shape == (2, len(sc.sspice_et), 3)
    assert np.array_equal(et, sc.sspice_et)
    assert np.array_equal(gps, sc.sspice_gps)
    assert np.array_equal(utc, sc.sc_utc)
    for i, body in enumerate(BODIES):
        direct = spice.spkpos(body, sc.sspice_et, "ITRF93", "NONE", "EARTH")[0] * 1000
        assert np.array_equal(r[i], direct)


# ---------------------------------------------------------------------------
# EP-04 cache behaviour
# ---------------------------------------------------------------------------
def test_ep04_repeated_times_no_extra_provider_calls():
    """EP-04: asking again for cached times does not call the provider.

    Expected: 1 provider call after the first request, still 1 after repeats,
    and identical results each time.
    """
    prov = FakeProvider()
    env = _env(prov)
    gps = np.arange(0.0, 100.0, 10.0)

    first = _get(env, gps)
    assert prov.calls == 1
    second = _get(env, gps)
    third = _get(env, gps[3:7][::-1])  # reordered strict subset is also cached

    assert prov.calls == 1
    assert np.array_equal(first, second)
    assert np.array_equal(first[:, 3:7][:, ::-1], third)
    assert first.shape == (2, 10, 3)


def test_ep04_only_missing_bins_requested():
    """EP-04: a partially new time set requests only the uncached epochs.

    Expected: the second provider call receives exactly the new ET values
    (from ``FakeProvider.requested``), and the combined result matches a fresh
    environment asked for everything at once.
    """
    prov = FakeProvider()
    env = _env(prov)
    old = np.arange(0.0, 50.0, 10.0)
    new = np.arange(50.0, 80.0, 10.0)

    _get(env, old)
    both = _get(env, np.concatenate([old[2:], new]))

    assert prov.calls == 2
    assert np.array_equal(prov.requested[1], new + ET_OFFSET)
    fresh = _get(_env(FakeProvider()), np.concatenate([old[2:], new]))
    assert np.array_equal(both, fresh)


def test_ep04_tolerance_binning():
    """EP-04 / F21: times inside one tolerance bin share one cache entry.

    Expected: bins are ``round(gps / tol)``. With tol=0.5, 100.0 and 100.2 both
    map to bin 200 (one cache entry); 100.3 maps to bin 201 (a new entry). A
    later request for another time in a cached bin makes no provider call.

    Contract (R18): an uncached bin is requested once, represented by the
    first time in request order (100.0), whatever later times share the bin;
    the bin then holds the position at 100.0.
    """
    prov = FakeProvider()
    env = _env(prov, tolerance=0.5)

    r = _get(env, [100.0, 100.2])
    assert prov.calls == 1
    # only the first time of the bin is requested ...
    assert np.array_equal(prov.requested[0], np.array([100.0]) + ET_OFFSET)
    assert set(env._cache) == {200}
    # ... and the shared bin holds its value (100.0)
    expected = FakeProvider()(BODIES, ephem_time=np.array([100.0 + ET_OFFSET]))
    assert np.array_equal(env._cache[200], expected[:, 0])
    assert np.array_equal(r[:, 0], r[:, 1])  # same bin -> same stored position
    assert np.array_equal(r[:, 0], expected[:, 0])

    _get(env, [99.8])  # bin 200 again
    assert prov.calls == 1
    assert np.array_equal(env._cache[200], expected[:, 0])  # unchanged
    _get(env, [100.3])  # bin 201
    assert prov.calls == 2
    assert set(env._cache) == {200, 201}


def test_ep04_bin_first_requested_time_represents_bin():
    """F21: the first-requested time of a bin represents it, in any order.

    Expected: in reverse order (100.2 then 100.0) only 100.2 is requested and
    the bin holds the position at 100.2. Several bins with interleaved
    duplicates: one provider call, each bin requested once, in first-seen
    order, and every returned time carries its bin's value.
    """
    prov = FakeProvider()
    env = _env(prov, tolerance=0.5)
    r = _get(env, [100.2, 100.0])
    assert np.array_equal(prov.requested[0], np.array([100.2]) + ET_OFFSET)
    expected = FakeProvider()(BODIES, ephem_time=np.array([100.2 + ET_OFFSET]))
    assert np.array_equal(r, np.repeat(expected, 2, axis=1))

    prov = FakeProvider()
    env = _env(prov, tolerance=0.5)
    gps = [10.1, 20.0, 10.0, 9.9, 20.2, 30.0, 19.8]  # bins 20, 40, 20, 20, 40, 60, 40
    r = _get(env, gps)
    assert prov.calls == 1
    firsts = np.array([10.1, 20.0, 30.0])
    assert np.array_equal(prov.requested[0], firsts + ET_OFFSET)
    assert set(env._cache) == {20, 40, 60}
    want = FakeProvider()(BODIES, ephem_time=firsts + ET_OFFSET)
    assert np.array_equal(r, want[:, [0, 1, 0, 0, 1, 2, 1]])


def test_ep04_tolerance_none_exact_times():
    """EP-04: ``tolerance=None`` keys the cache on the exact GPS value.

    Expected: tiny time differences are distinct entries; ``env.tolerance`` is
    None; the identical time is served from the cache.
    """
    prov = FakeProvider()
    env = _env(prov, tolerance=None)
    assert env.tolerance is None

    _get(env, [100.0, 100.0 + 1e-6])
    assert prov.calls == 1
    assert len(env._cache) == 2
    _get(env, [100.0])
    assert prov.calls == 1


def test_ep04_tolerance_zero_integer_second_bins():
    """EP-04: ``tolerance=0.0`` bins to the nearest whole GPS second.

    Expected (``_quantize``: ``round(gps)``): 100.2 and 100.4 share bin 100,
    100.6 is bin 101.
    """
    prov = FakeProvider()
    env = _env(prov, tolerance=0.0)
    assert env.tolerance == 0.0

    _get(env, [100.2, 100.4])
    assert set(env._cache) == {100}
    assert prov.calls == 1
    _get(env, [100.6])
    assert set(env._cache) == {100, 101}
    assert prov.calls == 2


def test_ep04_single_epoch_shape():
    """EP-04: N=1 input works and returns shape (B,1,3).

    Expected: shape ``(2, 1, 3)`` on the first (uncached) and second (cached)
    request, with equal values.
    """
    prov = FakeProvider()
    env = _env(prov)

    r1 = _get(env, 123.0)
    r2 = _get(env, 123.0)

    assert r1.shape == (2, 1, 3)
    assert r2.shape == (2, 1, 3)
    assert prov.calls == 1
    assert np.array_equal(r1, r2)


def test_ep04_constructor_preloads_times():
    """EP-04: ``ephem_time``/``gps_time``/``utc_time`` kwargs preload the cache.

    Expected: exactly 1 provider call in the constructor and none for a later
    ``get_ephem`` on the same times.
    """
    prov = FakeProvider()
    gps = np.arange(0.0, 60.0, 10.0)
    et, g, utc = _times(gps)

    env = _env(prov, ephem_time=et, gps_time=g, utc_time=utc)

    assert prov.calls == 1
    assert len(env._cache) == 6
    r = _get(env, gps)
    assert prov.calls == 1
    assert r.shape == (2, 6, 3)


def test_ep04_no_preload_without_times():
    """EP-04: with no time kwargs the constructor makes no provider call."""
    prov = FakeProvider()
    _env(prov)
    assert prov.calls == 0


def test_ep04_get_ephem_returns_inputs_and_positions():
    """EP-04: ``get_ephem`` returns (ET, GPS, UTC, positions) with inputs unchanged.

    Expected: the first three outputs equal the inputs; positions match what
    ``FakeProvider`` computes for the same ET (analytic cosine/sine of ET).
    """
    et, g, utc = _times(np.arange(0.0, 30.0, 10.0))
    env = _env(FakeProvider())

    et_o, g_o, utc_o, r = env.get_ephem(ephem_time=et, gps_time=g, utc_time=utc)

    assert np.array_equal(et_o, et)
    assert np.array_equal(g_o, g)
    assert np.array_equal(utc_o, utc)
    assert np.array_equal(r, FakeProvider()(BODIES, ephem_time=et))


# ---------------------------------------------------------------------------
# EP-05 GM lookup
# ---------------------------------------------------------------------------
def _prov(ephemeris: str) -> FakeProvider:
    """``FakeProvider`` reporting the given ephemeris name."""
    prov = FakeProvider()
    prov.ephemeris = ephemeris
    return prov


@pytest.mark.parametrize("name", ["de440", "de440s"])
def test_ep05_gm_from_provider_ephemeris(name):
    """EP-05: GM comes from ``GMc[provider.ephemeris[:5]]`` in body order.

    Expected values are ``constants.GMc['de440']`` (JPL DE440 GM, m^3/s^2) for
    both 'de440' and 'de440s'; lower-case body names are upper-cased.
    """
    prov = _prov(name)
    env = SolarSystemEnvironment(["moon", "Sun", "EARTH"], 0.001, prov)

    assert prov.calls == 0  # GM lookup alone never calls the provider
    gm = GMc["de440"]
    assert env.bodies == ["MOON", "SUN", "EARTH"]
    assert np.array_equal(env.GM, [gm["MOON"], gm["SUN"], gm["EARTH"]])
    assert env.GM.dtype == np.float64


def test_ep05_gm_uses_provider_ephemeris_key():
    """EP-05: the GM table is selected by the provider's ephemeris name.

    Expected: an ephemeris with no entry in ``GMc`` raises a ``ValueError``
    naming the ephemeris and listing the known ones, before any provider call.
    """
    prov = _prov("de999")
    with pytest.raises(ValueError, match=r"'de999'.*\['de440'\]"):
        SolarSystemEnvironment(BODIES, 0.001, prov, *_times([0.0]))
    assert prov.calls == 0


def test_ep05_unknown_body_value_error():
    """EP-05: an unknown body raises a clear ``ValueError``.

    Expected: the message names the unknown body and lists the known bodies
    of that ephemeris (``unit`` metadata excluded); no provider call is made,
    even when preload times are given.
    """
    prov = FakeProvider()
    with pytest.raises(ValueError) as exc:
        SolarSystemEnvironment(["SUN", "notabody"], 0.001, prov, *_times([0.0]))
    msg = str(exc.value)
    assert "NOTABODY" in msg and "de440s" in msg
    for body in ("SUN", "MOON", "EARTH", "PLUTO"):
        assert f"'{body}'" in msg
    assert "unit" not in msg
    assert prov.calls == 0
