"""Tests for SPICE kernel lookup, download location and loading (R16b).

No network: ``dl_file`` and ``wf_mtime`` are replaced by fakes, and the SPICE
kernel pool is replaced by :class:`FakePool` wherever kernels would be loaded
or unloaded, so the real pool used by the rest of the suite is not touched.
Every test points ``CONTIGO_CACHE_DIR`` at its own temp dir.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests
import spiceypy as spice

import contigo.config as config
from contigo.contigo_utils import kernels, utils
from contigo.contigo_utils.time_utils import check_lpsk
from contigo.contigo_utils.utils import dl_file as _real_dl_file
from contigo.ephemeris.spice_ephem import SPICEEphem

NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
FILES = ("de440s.bsp", config.LEAP_FILE, config.PCK_FILE)
URLS = {
    "de440s.bsp": f"{NAIF}/spk/planets/de440s.bsp",
    config.LEAP_FILE: f"{NAIF}/lsk/{config.LEAP_FILE}",
    config.PCK_FILE: f"{NAIF}/pck/{config.PCK_FILE}",
}
OLD = datetime(2000, 1, 1, tzinfo=timezone.utc)
NEW = datetime(2099, 1, 1, tzinfo=timezone.utc)  # NAIF copy is newer


class FakePool:
    """Stand-in for the SPICE kernel pool; records furnsh/unload calls."""

    def __init__(self, loaded=()):
        self.files = [str(f) for f in loaded]
        self.calls: list[tuple[str, str]] = []

    def install(self, mp: pytest.MonkeyPatch) -> FakePool:
        mp.setattr(spice, "ktotal", lambda kind: len(self.files))
        mp.setattr(spice, "kdata", lambda i, kind: (self.files[i], "", "", 0))
        mp.setattr(spice, "furnsh", self._furnsh)
        mp.setattr(spice, "unload", self._unload)
        return self

    def _furnsh(self, path):
        self.calls.append(("furnsh", path))
        self.files.append(path)

    def _unload(self, path):
        self.calls.append(("unload", path))
        self.files.remove(path)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Empty fake ``DATA_DIR`` and cache dir; online unless a test says so."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    data.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", str(data))
    monkeypatch.setenv("CONTIGO_CACHE_DIR", str(cache))
    monkeypatch.delenv("CONTIGO_OFFLINE", raising=False)
    return data, cache / "spice"


@pytest.fixture
def fake_dl(monkeypatch):
    """``dl_file`` that writes a tiny file and records ``(url, target)``.

    ``kernels._download`` writes to ``<target>.<pid>.part`` and then moves it
    into place, so the final ``target`` is recorded.
    """
    downloads: list[tuple[str, str]] = []

    def _dl(url, filename):
        downloads.append((url, re.sub(r"\.\d+\.part$", "", filename)))
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_text("downloaded")

    monkeypatch.setattr(utils, "dl_file", _dl)
    return downloads


def _old_file(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("local")
    t = OLD.timestamp()
    os.utime(p, (t, t))
    return p


# ---------------------------------------------------------------------------
# NAIF URL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FILES)
def test_naif_url_from_kernel_type(name):
    assert kernels.naif_url(name) == URLS[name]


def test_naif_url_unknown_type_raises():
    with pytest.raises(ValueError, match="kernel type"):
        kernels.naif_url("kernel.xyz")


# ---------------------------------------------------------------------------
# Download location and resolution
# ---------------------------------------------------------------------------
def test_dl_kernels_downloads_missing_into_cache(dirs, fake_dl, monkeypatch):
    data, cache = dirs

    def _no_mtime(url):
        raise AssertionError("wf_mtime called for a missing kernel")

    monkeypatch.setattr(utils, "wf_mtime", _no_mtime)
    config.state["kernel_downloaded"] = False

    paths = SPICEEphem.__new__(SPICEEphem).dl_kernels(*FILES)

    assert fake_dl == [(URLS[f], str(cache / f)) for f in FILES]
    assert paths == [cache / f for f in FILES]
    assert all(p.read_text() == "downloaded" for p in paths)
    assert os.listdir(data) == []
    assert config.state["kernel_downloaded"] is True


def test_newer_naif_copy_goes_to_cache_bundled_untouched(dirs, fake_dl, monkeypatch):
    data, cache = dirs
    bundled = _old_file(data / config.PCK_FILE)
    monkeypatch.setattr(utils, "wf_mtime", lambda url: datetime.now(timezone.utc))

    path = kernels.ensure_kernel(config.PCK_FILE)

    assert path == cache / config.PCK_FILE
    assert fake_dl == [(URLS[config.PCK_FILE], str(path))]
    assert bundled.read_text() == "local"
    # the new download is now what the lookup finds
    assert config.find_data_file(config.PCK_FILE, "spice") == path


def test_older_naif_copy_keeps_local_file(dirs, fake_dl, monkeypatch):
    data, _ = dirs
    bundled = _old_file(data / config.PCK_FILE)
    monkeypatch.setattr(
        utils, "wf_mtime", lambda url: datetime(1990, 1, 1, tzinfo=timezone.utc)
    )

    assert kernels.ensure_kernel(config.PCK_FILE) == bundled
    assert fake_dl == []


def test_cached_kernel_takes_precedence_over_bundled(dirs, fake_dl):
    data, cache = dirs
    _old_file(data / config.LEAP_FILE)
    cached = _old_file(cache / config.LEAP_FILE)

    assert kernels.ensure_kernel(config.LEAP_FILE, check_remote=False) == cached
    assert fake_dl == []


def test_check_lpsk_downloads_into_cache_with_naif_url(dirs, fake_dl, monkeypatch):
    data, cache = dirs
    pool = FakePool().install(monkeypatch)

    check_lpsk()

    target = str(cache / config.LEAP_FILE)
    assert fake_dl == [(kernels.naif_url(config.LEAP_FILE), target)]
    assert pool.calls == [("furnsh", target)]
    assert os.listdir(data) == []


def test_real_data_dir_unchanged_by_downloads(tmp_path, fake_dl, monkeypatch):
    """Guard: download paths never write into the real bundled ``DATA_DIR``.

    A kernel missing from ``DATA_DIR`` (de440.bsp) and bundled kernels with a
    newer NAIF copy are all downloaded into the cache.
    """
    monkeypatch.setenv("CONTIGO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CONTIGO_OFFLINE", raising=False)
    monkeypatch.setattr(utils, "wf_mtime", lambda url: datetime.now(timezone.utc))
    before = sorted(os.listdir(config.DATA_DIR))

    paths = SPICEEphem.__new__(SPICEEphem).dl_kernels(
        "de440.bsp", config.LEAP_FILE, config.PCK_FILE
    )

    assert len(fake_dl) == 3
    assert all(p.parent == tmp_path / "cache" / "spice" for p in paths)
    assert sorted(os.listdir(config.DATA_DIR)) == before


# ---------------------------------------------------------------------------
# Offline
# ---------------------------------------------------------------------------
def test_offline_no_freshness_check_and_missing_raises(dirs, monkeypatch):
    data, _ = dirs
    monkeypatch.setenv("CONTIGO_OFFLINE", "1")
    monkeypatch.setattr(utils, "dl_file", _real_dl_file)

    def _no_mtime(url):
        raise AssertionError("wf_mtime called while offline")

    monkeypatch.setattr(utils, "wf_mtime", _no_mtime)
    local = _old_file(data / config.LEAP_FILE)

    assert kernels.ensure_kernel(config.LEAP_FILE) == local
    with pytest.raises(config.OfflineError, match=config.PCK_FILE):
        kernels.ensure_kernel(config.PCK_FILE)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_same_file_via_other_spelling_is_already_loaded(tmp_path, monkeypatch):
    k = tmp_path / "sub" / config.LEAP_FILE
    k.parent.mkdir()
    k.write_text("x")
    other = str(tmp_path / "sub" / ".." / "sub" / config.LEAP_FILE)
    pool = FakePool([other]).install(monkeypatch)

    assert kernels.furnsh_kernel(k) is False
    assert pool.calls == []


@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive paths")
def test_same_file_differing_in_case_and_slashes_is_already_loaded(
    tmp_path, monkeypatch
):
    k = tmp_path / config.LEAP_FILE
    k.write_text("x")
    other = str(k).upper().replace("\\", "/")
    pool = FakePool([other]).install(monkeypatch)

    assert kernels.furnsh_kernel(k) is False
    assert pool.calls == []


def test_same_name_other_path_unloads_old_first(tmp_path, monkeypatch):
    old = tmp_path / "data" / config.PCK_FILE
    new = tmp_path / "cache" / config.PCK_FILE
    for p in (old, new):
        p.parent.mkdir()
        p.write_text("x")
    pool = FakePool(["/x/other.bsp", str(old)]).install(monkeypatch)

    assert kernels.furnsh_kernel(new) is True
    assert pool.calls == [("unload", str(old)), ("furnsh", str(new))]
    assert pool.files == ["/x/other.bsp", str(new)]


def test_load_kernels_replaces_bundled_with_newer_download(dirs, fake_dl, monkeypatch):
    """End to end: bundled kernels loaded, a newer PCK downloaded to the cache."""
    data, cache = dirs
    bundled = [_old_file(data / f) for f in FILES]
    pool = FakePool(bundled).install(monkeypatch)
    newer = {URLS[config.PCK_FILE]}
    monkeypatch.setattr(
        utils,
        "wf_mtime",
        lambda url: datetime.now(timezone.utc) if url in newer else OLD,
    )
    config.state["kernel_downloaded"] = False

    sp = SPICEEphem.__new__(SPICEEphem)
    sp.ephemeris = "de440s"
    sp.load_kernels()

    pck_new = str(cache / config.PCK_FILE)
    assert fake_dl == [(URLS[config.PCK_FILE], pck_new)]
    assert pool.calls == [
        ("unload", str(data / config.PCK_FILE)),
        ("furnsh", pck_new),
    ]
    # second call: nothing to download (flag set), nothing to load
    pool.calls.clear()
    sp.load_kernels()
    assert pool.calls == []
    assert len(fake_dl) == 1


# ---------------------------------------------------------------------------
# Atomic downloads (F32): a failed download never replaces the cached kernel
# ---------------------------------------------------------------------------
def test_failed_download_keeps_cached_kernel(dirs, monkeypatch, caplog):
    """A refresh cut off part-way leaves the cached kernel, no ``.part`` file,
    and ``ensure_kernel`` falls back to the cached copy with a warning."""
    _, cache = dirs
    cached = _old_file(cache / config.PCK_FILE)
    monkeypatch.setattr(utils, "wf_mtime", lambda url: NEW)

    def cut_off(url, filename):
        Path(filename).write_bytes(b"trunc")
        raise requests.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr(utils, "dl_file", cut_off)
    with caplog.at_level("WARNING", logger="contigo.contigo_utils.kernels"):
        assert kernels.ensure_kernel(config.PCK_FILE) == cached
    assert "connection reset" in caplog.text
    assert cached.read_text() == "local"
    assert list(cache.iterdir()) == [cached]


def test_failed_download_of_missing_kernel_raises(dirs, monkeypatch):
    """With no local copy to fall back to, the download error is raised."""
    _, cache = dirs

    def cut_off(url, filename):
        Path(filename).write_bytes(b"trunc")
        raise requests.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr(utils, "dl_file", cut_off)
    with pytest.raises(requests.exceptions.ConnectionError):
        kernels.ensure_kernel(config.PCK_FILE)
    assert not cache.exists() or list(cache.iterdir()) == []


def test_short_download_is_rejected(dirs, monkeypatch):
    """A download shorter than ``Content-Length`` raises ``OSError`` and the
    cached kernel is kept."""
    _, cache = dirs
    cached = _old_file(cache / config.PCK_FILE)

    def short(url, filename):
        Path(filename).write_bytes(b"12345")
        return 10  # what the server announced

    monkeypatch.setattr(utils, "dl_file", short)
    with pytest.raises(OSError, match="5 of 10 bytes"):
        kernels._download("https://example.invalid/k.bpc", cached)
    assert cached.read_text() == "local"
    assert list(cache.iterdir()) == [cached]


def test_failed_replace_reloads_the_old_kernel(dirs, monkeypatch):
    """If moving the download into place fails (file locked), the kernel that
    was unloaded for the swap is loaded again."""
    _, cache = dirs
    cached = _old_file(cache / config.PCK_FILE)
    pool = FakePool([cached]).install(monkeypatch)
    monkeypatch.setattr(utils, "dl_file", lambda url, fn: Path(fn).write_bytes(b"x"))

    def locked(src, dst):
        raise PermissionError("file in use")

    monkeypatch.setattr(kernels.os, "replace", locked)
    with pytest.raises(PermissionError):
        kernels._download("https://example.invalid/k.bpc", cached)
    assert pool.calls == [("unload", str(cached)), ("furnsh", str(cached))]
    assert cached.read_text() == "local"
    assert list(cache.iterdir()) == [cached]


def test_complete_download_replaces_cached_kernel(dirs, monkeypatch):
    """A complete download (matching size) replaces the cached copy."""
    _, cache = dirs
    cached = _old_file(cache / config.PCK_FILE)
    monkeypatch.setattr(utils, "wf_mtime", lambda url: NEW)

    def ok(url, filename):
        Path(filename).write_bytes(b"new kernel")
        return 10

    monkeypatch.setattr(utils, "dl_file", ok)
    assert kernels.ensure_kernel(config.PCK_FILE) == cached
    assert cached.read_bytes() == b"new kernel"
    assert list(cache.iterdir()) == [cached]
