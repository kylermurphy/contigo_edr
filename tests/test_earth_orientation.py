"""Tests for ``kernels.check_pck`` and the PCK coverage helpers (EO-01 .. EO-08).

Expected values: the bundled ``earth_latest_high_prec.bpc`` covers ITRF93
(SPICE body 3000) from 2000-01-01 to 2026-05-30 (``spice.pckcov``). No test
uses the network: ``CONTIGO_OFFLINE`` or fakes for ``dl_file``/``wf_mtime``.

added: 10/11/2026 Claude Code
"""

from __future__ import annotations

import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import requests
import spiceypy as spice

import contigo.config as config
from contigo.contigo_utils import kernels, utils

ET_2018 = 5.68123251e8  # inside the bundled coverage
ET_2026_09 = 8.4e8  # after the bundled PCK ends


@pytest.fixture
def fresh_session(monkeypatch, tmp_path):
    """Empty cache dir and "no NAIF check done yet" flags, as in a new process."""
    monkeypatch.setenv("CONTIGO_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(config.state, "pck_refreshed", False)
    monkeypatch.setitem(config.state, "kernel_downloaded", False)


@pytest.fixture
def spy_ensure(monkeypatch):
    """Record ``ensure_kernel`` calls (name, check_remote) and run the real one."""
    calls: list[tuple[str, bool]] = []
    real = kernels.ensure_kernel

    def spy(name, check_remote=True):
        calls.append((name, check_remote))
        return real(name, check_remote=check_remote)

    monkeypatch.setattr(kernels, "ensure_kernel", spy)
    return calls


def test_eo01_bundled_pck_coverage():
    """EO-01: the bundled PCK covers 2018 but not 2026-09."""
    kernels.check_pck(None)
    assert kernels.pck_covers(ET_2018, ET_2018 + 86400)
    assert not kernels.pck_covers(ET_2018, ET_2026_09)


def test_eo02_covered_epochs_never_touch_the_network(fresh_session, monkeypatch):
    """EO-02: covered epochs return ITRF93 without any NAIF request."""

    def fail(*args, **kwargs):
        pytest.fail("network used")

    monkeypatch.setattr(utils, "wf_mtime", fail)
    monkeypatch.setattr(utils, "dl_file", fail)
    assert kernels.check_pck([ET_2018]) == "ITRF93"
    assert kernels.check_pck(None) == "ITRF93"


def test_eo03_uncovered_offline_warns_and_retries_once(
    fresh_session, spy_ensure, monkeypatch
):
    """EO-03: uncovered + offline -> RuntimeWarning + IAU_EARTH every call, but
    only one attempt to update the PCK per session."""
    monkeypatch.setenv("CONTIGO_OFFLINE", "1")
    for _ in range(3):
        with pytest.warns(RuntimeWarning, match="IAU_EARTH"):
            assert kernels.check_pck([ET_2026_09]) == "IAU_EARTH"
    assert spy_ensure.count((config.PCK_FILE, True)) == 1
    assert config.state["pck_refreshed"] is True


def test_eo04_raise_and_bad_argument(fresh_session, monkeypatch):
    """EO-04: ``fallback='raise'`` names the covered range; bad value rejected."""
    monkeypatch.setenv("CONTIGO_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="2026-05-30"):
        kernels.check_pck([ET_2026_09], fallback="raise")
    with pytest.raises(ValueError, match="fallback"):
        kernels.check_pck([ET_2018], fallback="ignore")


def test_eo05_update_that_covers_the_epochs_gives_itrf93(
    fresh_session, monkeypatch, spy_ensure
):
    """EO-05: if the update makes the epochs covered: ITRF93 and no warning.

    Online with a fake NAIF copy dated far in the future; ``pck_covers`` is
    faked to say "covered" once the refresh (``check_remote=True``) has run.
    """
    monkeypatch.delenv("CONTIGO_OFFLINE", raising=False)
    monkeypatch.setattr(
        utils, "wf_mtime", lambda url: datetime(2099, 1, 1, tzinfo=timezone.utc)
    )
    downloads = []

    def fake_dl(url, filename):
        downloads.append(url)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"x")

    monkeypatch.setattr(utils, "dl_file", fake_dl)
    # the downloaded stub is not a real kernel: do not furnish it
    monkeypatch.setattr(kernels, "furnsh_kernel", lambda path: False)
    monkeypatch.setattr(
        kernels, "pck_covers", lambda a, b: (config.PCK_FILE, True) in spy_ensure
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert kernels.check_pck([ET_2026_09]) == "ITRF93"
    assert len(downloads) == 1 and downloads[0].endswith(config.PCK_FILE)


def test_eo06_no_second_update_after_spiceephem_check(
    fresh_session, spy_ensure, monkeypatch
):
    """EO-06: if SPICEEphem already checked NAIF this session
    (``kernel_downloaded``), ``check_pck`` does not ask again."""
    monkeypatch.setitem(config.state, "kernel_downloaded", True)
    with pytest.warns(RuntimeWarning):
        kernels.check_pck([ET_2026_09])
    assert (config.PCK_FILE, True) not in spy_ensure


def test_eo07_fallback_frame_needs_the_bundled_text_pck():
    """EO-07: the bundled ``pck00011.tpc`` makes IAU_EARTH usable (F28)."""
    kernels.furnsh_kernel(kernels.ensure_kernel(config.LEAP_FILE, check_remote=False))
    kernels.furnsh_kernel(kernels.ensure_kernel(config.TPC_FILE, check_remote=False))
    rot = spice.pxform("IAU_EARTH", "J2000", ET_2018)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-12)
    assert not np.allclose(rot, np.eye(3))  # a real rotation, not the identity


def test_eo08_cut_off_refresh_keeps_the_cached_pck_usable(
    fresh_session, tmp_path, monkeypatch
):
    """EO-08 (F32): a PCK refresh that fails part-way leaves the cached PCK
    intact and loaded, so covered epochs still get ITRF93 afterwards.

    Before the fix the cached file was unloaded and then overwritten by the
    partial download, and the next ``furnsh`` failed on every later run.
    """
    monkeypatch.delenv("CONTIGO_OFFLINE", raising=False)
    cached = tmp_path / "spice" / config.PCK_FILE
    cached.parent.mkdir()
    shutil.copyfile(kernels.ensure_kernel(config.PCK_FILE, check_remote=False), cached)
    original = cached.read_bytes()
    monkeypatch.setattr(
        utils, "wf_mtime", lambda url: datetime(2099, 1, 1, tzinfo=timezone.utc)
    )

    def cut_off(url, filename):
        Path(filename).write_bytes(b"DAF/PCK truncated")
        raise requests.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr(utils, "dl_file", cut_off)
    try:
        with pytest.warns(RuntimeWarning, match="IAU_EARTH"):
            assert kernels.check_pck([ET_2026_09]) == "IAU_EARTH"
        assert cached.read_bytes() == original
        assert kernels.check_pck([ET_2018]) == "ITRF93"
        assert sorted(p.name for p in cached.parent.iterdir()) == [config.PCK_FILE]
    finally:
        kernels._unload_path(cached)  # the next test reloads the bundled copy
