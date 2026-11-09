"""Tests for the offline switch, cache dir and data lookup (R11, R16a, R4).

``dl_file`` and ``wf_mtime`` are replaced by the autouse ``_no_network``
fixture, so the originals are captured at import time (as in test_utils.py).

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests

import contigo.config as config
from contigo.contigo_utils import utils
from contigo.contigo_utils.utils import dl_file as _real_dl_file
from contigo.contigo_utils.utils import wf_mtime as _real_wf_mtime
from contigo.ephemeris.spice_ephem import SPICEEphem

URL = "https://example.invalid/file.bin"


@pytest.fixture
def no_requests(monkeypatch):
    """Fail if anything in ``requests`` is called."""

    def _fail(*args, **kwargs):
        raise AssertionError("a request was made while offline")

    for name in ("get", "head"):
        monkeypatch.setattr(requests, name, _fail)
        monkeypatch.setattr(utils.requests, name, _fail)


# ---------------------------------------------------------------------------
# is_offline / OfflineError
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " ON "])
def test_is_offline_truthy(monkeypatch, value):
    monkeypatch.setenv("CONTIGO_OFFLINE", value)
    assert config.is_offline() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "2"])
def test_is_offline_falsy(monkeypatch, value):
    monkeypatch.setenv("CONTIGO_OFFLINE", value)
    assert config.is_offline() is False


def test_is_offline_unset(monkeypatch):
    monkeypatch.delenv("CONTIGO_OFFLINE", raising=False)
    assert config.is_offline() is False


def test_offline_error_is_runtime_error():
    assert issubclass(config.OfflineError, RuntimeError)


def test_dl_file_offline_raises_without_request(monkeypatch, tmp_path, no_requests):
    monkeypatch.setenv("CONTIGO_OFFLINE", "1")
    with pytest.raises(config.OfflineError, match="example.invalid"):
        _real_dl_file(URL, tmp_path / "f.bin")
    assert not (tmp_path / "f.bin").exists()


def test_wf_mtime_offline_returns_none_without_request(monkeypatch, no_requests):
    monkeypatch.setenv("CONTIGO_OFFLINE", "1")
    assert _real_wf_mtime(URL) is None


# ---------------------------------------------------------------------------
# cache_dir
# ---------------------------------------------------------------------------
def test_cache_dir_env_override_and_subdirs(monkeypatch, tmp_path):
    root = tmp_path / "cache"
    monkeypatch.setenv("CONTIGO_CACHE_DIR", str(root))
    assert config.cache_dir() == root
    assert root.is_dir()
    for sub in ("spice", "gravity", "orekit"):
        d = config.cache_dir(sub)
        assert d == root / sub
        assert d.is_dir()


def test_cache_dir_default_uses_platformdirs(monkeypatch, tmp_path):
    calls = []

    def fake_user_cache_dir(appname, appauthor=None, *args, **kwargs):
        calls.append((appname, appauthor))
        return str(tmp_path / "pd_cache")

    monkeypatch.delenv("CONTIGO_CACHE_DIR", raising=False)
    monkeypatch.setattr(config.platformdirs, "user_cache_dir", fake_user_cache_dir)
    assert config.cache_dir("spice") == tmp_path / "pd_cache" / "spice"
    assert (tmp_path / "pd_cache" / "spice").is_dir()
    assert calls == [("contigo", False)]


def test_session_cache_dir_is_isolated():
    # the conftest fixture points the cache away from the real user cache
    import platformdirs

    real = Path(platformdirs.user_cache_dir("contigo", appauthor=False))
    assert config.cache_dir() != real


# ---------------------------------------------------------------------------
# DATA_DIR / data_path
# ---------------------------------------------------------------------------
def test_data_dir_is_bundled_package_data():
    assert config.DATA_DIR == config.data_path()
    assert (Path(config.DATA_DIR) / config.LEAP_FILE).is_file()
    assert Path(config.DATA_DIR).parent == Path(config.__file__).resolve().parent


def test_data_path_creates_nothing(monkeypatch):
    def _no_mkdir(*args, **kwargs):
        raise AssertionError("data_path must not create directories")

    monkeypatch.setattr(os, "makedirs", _no_mkdir)
    monkeypatch.setattr(os, "mkdir", _no_mkdir)
    monkeypatch.setattr(Path, "mkdir", _no_mkdir)
    config.data_path()


@pytest.mark.parametrize(
    "files",
    [
        lambda pkg: Path("/definitely/not/a/dir"),
        lambda pkg: (_ for _ in ()).throw(RuntimeError("zip import")),
    ],
    ids=["not-a-dir", "raises"],
)
def test_data_path_falls_back_to_module_dir(monkeypatch, files):
    monkeypatch.setattr(config, "_files", files)
    expected = (Path(config.__file__).parent / "data").resolve()
    assert config.data_path() == str(expected)


# ---------------------------------------------------------------------------
# find_data_file
# ---------------------------------------------------------------------------
@pytest.fixture
def lookup_dirs(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("CONTIGO_CACHE_DIR", str(cache))
    monkeypatch.setattr(config, "DATA_DIR", str(bundled))
    return tmp_path, cache / "gravity", bundled


def test_find_data_file_order(lookup_dirs):
    tmp_path, cache, bundled = lookup_dirs
    cache.mkdir(parents=True)
    explicit = tmp_path / "elsewhere" / "f.gfc"
    explicit.parent.mkdir()
    for p in (explicit, cache / "f.gfc", bundled / "f.gfc"):
        p.write_text("x")

    assert config.find_data_file(explicit, "gravity") == explicit.resolve()
    assert config.find_data_file("f.gfc", "gravity") == cache / "f.gfc"
    (cache / "f.gfc").unlink()
    assert config.find_data_file("f.gfc", "gravity") == bundled / "f.gfc"


def test_find_data_file_missing_explicit_path_does_not_fall_back(lookup_dirs):
    tmp_path, _, bundled = lookup_dirs
    (bundled / "f.gfc").write_text("x")
    assert config.find_data_file(Path("nope") / "f.gfc", "gravity") is None
    assert config.find_data_file("nope/f.gfc", "gravity") is None
    assert config.find_data_file(tmp_path / "missing" / "f.gfc", "gravity") is None


def test_find_data_file_relative_path_is_resolved(lookup_dirs, monkeypatch):
    tmp_path, _, bundled = lookup_dirs
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "rel.gfc").write_text("x")
    (tmp_path / "rel.gfc").write_text("x")
    (bundled / "rel.gfc").write_text("x")
    monkeypatch.chdir(tmp_path)
    found = config.find_data_file(Path("sub") / "rel.gfc", "gravity")
    assert found == (tmp_path / "sub" / "rel.gfc").resolve()
    assert found.is_absolute()
    assert (
        config.find_data_file("rel.gfc", "gravity") == (tmp_path / "rel.gfc").resolve()
    )


def test_find_data_file_missing(lookup_dirs):
    assert config.find_data_file("missing.gfc", "gravity") is None


def test_find_data_file_creates_nothing(lookup_dirs, monkeypatch):
    tmp_path, cache, bundled = lookup_dirs
    (bundled / "f.gfc").write_text("x")

    def _no_mkdir(*args, **kwargs):
        raise PermissionError("read-only home")

    monkeypatch.setattr(Path, "mkdir", _no_mkdir)
    assert config.find_data_file("f.gfc", "gravity") == bundled / "f.gfc"
    assert not cache.parent.exists()


def test_cache_dir_create_false(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTIGO_CACHE_DIR", str(tmp_path / "c"))
    assert config.cache_dir("spice", create=False) == tmp_path / "c" / "spice"
    assert not (tmp_path / "c").exists()


# ---------------------------------------------------------------------------
# Logging (R4)
# ---------------------------------------------------------------------------
def test_contigo_logger_has_no_handlers():
    # a NullHandler would suppress the lastResort handler (warnings lost)
    assert logging.getLogger("contigo").handlers == []


def test_import_does_not_touch_root_logger():
    code = (
        "import logging; before = list(logging.getLogger().handlers); "
        "level = logging.getLogger().level; "
        "import contigo, contigo.config, contigo.contigo_utils.utils; "
        "assert logging.getLogger().handlers == before, logging.getLogger().handlers; "
        "assert logging.getLogger().level == level; "
        "assert logging.getLogger('contigo').handlers == []"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_warnings_reach_stderr_without_logging_config():
    code = (
        "import logging, contigo.config, contigo.contigo_utils.utils; "
        "logging.getLogger('contigo.contigo_utils.utils').warning('dup-epoch-xyz'); "
        "logging.getLogger('contigo.contigo_utils.utils').info('info-xyz')"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert "dup-epoch-xyz" in res.stderr
    assert "info-xyz" not in res.stderr + res.stdout


def test_get_gitlab_sha_offline_raises_without_request(monkeypatch, no_requests):
    orekit_utils = pytest.importorskip("contigo.contigo_utils.orekit_utils")
    assert orekit_utils.requests.get is requests.get  # patched by no_requests
    monkeypatch.setenv("CONTIGO_OFFLINE", "1")
    with pytest.raises(config.OfflineError, match="gitlab.orekit.org"):
        orekit_utils.get_gitlab_sha()


# ---------------------------------------------------------------------------
# dl_kernels offline
# ---------------------------------------------------------------------------
def test_dl_kernels_offline(monkeypatch, tmp_path):
    for f in ("de440s.bsp", config.LEAP_FILE):
        (tmp_path / f).write_text("x")
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONTIGO_OFFLINE", "1")
    monkeypatch.setattr(utils, "dl_file", _real_dl_file)

    def _no_mtime(url):
        raise AssertionError("wf_mtime called while offline")

    monkeypatch.setattr(utils, "wf_mtime", _no_mtime)
    config.state["kernel_downloaded"] = False

    sp = SPICEEphem.__new__(SPICEEphem)
    with pytest.raises(config.OfflineError, match=config.PCK_FILE):
        sp.dl_kernels("de440s.bsp", config.LEAP_FILE, config.PCK_FILE)
    assert config.state["kernel_downloaded"] is False

    # all present: no network, flag set
    (tmp_path / config.PCK_FILE).write_text("x")
    sp.dl_kernels("de440s.bsp", config.LEAP_FILE, config.PCK_FILE)
    assert config.state["kernel_downloaded"] is True
