"""Orekit helper jar and Orekit data resolution (R5, rows OR-01..OR-05, PK-04).

OR-01..OR-05 are fast-tier: no JVM, no network. ``requests.get`` and
``utils.dl_file`` are replaced; the cache dir is a temp dir
(``CONTIGO_CACHE_DIR``) and ``config.DATA_DIR`` is patched to another temp
dir standing in for the legacy tracked zip location.

PK-04 (``orekit`` tier, no network) starts the JVM offline in a subprocess
from a zip that is already on disk.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

import contigo
import contigo.config as config
from contigo.contigo_utils import utils

orekit_utils = pytest.importorskip("contigo.contigo_utils.orekit_utils")

SHA_NEW = "abcdef0123456789abcdef0123456789abcdef01"


@pytest.fixture
def dirs(monkeypatch, tmp_path):
    """Fresh cache dir and DATA_DIR; returns ``(cache/orekit, data_dir)``."""
    cache_root = tmp_path / "cache"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("CONTIGO_CACHE_DIR", str(cache_root))
    monkeypatch.delenv("CONTIGO_OFFLINE", raising=False)
    monkeypatch.setattr(config, "DATA_DIR", str(data_dir))
    return config.cache_dir("orekit"), data_dir


def _zip(path: Path, age_s: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK\x03\x04 fake")
    t = time.time() - age_s
    os.utime(path, (t, t))
    return path


@pytest.fixture
def no_requests(monkeypatch):
    calls = []

    def _fail(*args, **kwargs):
        calls.append(args)
        raise AssertionError("a request was made")

    monkeypatch.setattr(requests, "get", _fail)
    monkeypatch.setattr(utils, "dl_file", _fail)
    return calls


class _Resp:
    def __init__(self, status=200, sha=SHA_NEW):
        self.status_code = status
        self._sha = sha

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)

    def json(self):
        return {"id": self._sha}


def _sha_ok(monkeypatch, sha=SHA_NEW):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(200, sha))


# ---------------------------------------------------------------------------
# OR-01 offline
# ---------------------------------------------------------------------------
def test_or01_offline_newest_by_mtime_part_ignored(monkeypatch, dirs, no_requests):
    cache, data = dirs
    _zip(cache / "orekit_data_11111111.zip", age_s=300)
    newest = _zip(data / "orekit_data_22222222.zip", age_s=100)
    _zip(cache / "orekit_data_33333333.zip.part", age_s=0)
    monkeypatch.setenv("CONTIGO_OFFLINE", "1")

    assert orekit_utils.resolve_orekit_data() == newest
    assert no_requests == []


def test_or01_offline_none_raises(monkeypatch, dirs, no_requests):
    cache, data = dirs
    _zip(cache / "orekit_data_33333333.zip.part", age_s=0)
    monkeypatch.setenv("CONTIGO_OFFLINE", "1")

    with pytest.raises(orekit_utils.OrekitDataNotFoundError) as exc:
        orekit_utils.resolve_orekit_data()
    msg = str(exc.value)
    assert str(cache) in msg and str(data) in msg and "CONTIGO_OFFLINE" in msg
    assert isinstance(exc.value, FileNotFoundError)
    assert no_requests == []


# ---------------------------------------------------------------------------
# OR-02 GitLab unreachable
# ---------------------------------------------------------------------------
def _conn_error(*args, **kwargs):
    raise requests.ConnectionError("gitlab down")


def test_or02_sha_connection_error_falls_back(monkeypatch, dirs):
    cache, _ = dirs
    local = _zip(cache / "orekit_data_11111111.zip", age_s=10)
    monkeypatch.setattr(requests, "get", _conn_error)

    with pytest.warns(RuntimeWarning, match="gitlab down"):
        assert orekit_utils.resolve_orekit_data() == local


def test_or02_sha_connection_error_no_local_raises_chained(monkeypatch, dirs):
    monkeypatch.setattr(requests, "get", _conn_error)

    with pytest.raises(orekit_utils.OrekitDataNotFoundError) as exc:
        orekit_utils.resolve_orekit_data()
    assert isinstance(exc.value.__cause__, requests.ConnectionError)


# ---------------------------------------------------------------------------
# OR-03 404
# ---------------------------------------------------------------------------
def test_or03_get_gitlab_sha_404_raises(monkeypatch):
    monkeypatch.delenv("CONTIGO_OFFLINE", raising=False)
    seen = {}

    def _get(url, **kwargs):
        seen.update(kwargs)
        return _Resp(404)

    monkeypatch.setattr(requests, "get", _get)
    with pytest.raises(requests.HTTPError, match="404"):
        orekit_utils.get_gitlab_sha(project_id=999, branch="nope")
    assert seen.get("timeout") == 10


# ---------------------------------------------------------------------------
# OR-04 cache hit, download, failed download
# ---------------------------------------------------------------------------
def test_or04_known_sha_no_download(monkeypatch, dirs):
    cache, _ = dirs
    known = _zip(cache / f"orekit_data_{SHA_NEW[:8]}.zip", age_s=1000)
    _sha_ok(monkeypatch)
    monkeypatch.setattr(
        utils, "dl_file", lambda *a: pytest.fail("download not expected")
    )

    assert orekit_utils.resolve_orekit_data() == known


def test_or04_new_sha_downloads_and_prunes_cache(monkeypatch, dirs):
    cache, data = dirs
    old = _zip(cache / "orekit_data_11111111.zip", age_s=100)
    legacy = _zip(data / "orekit_data_22222222.zip", age_s=100)
    _sha_ok(monkeypatch)
    calls = []

    def _dl(url, filename):
        calls.append((url, filename))
        assert filename.endswith(".zip.part")
        Path(filename).write_bytes(b"PK new")

    monkeypatch.setattr(utils, "dl_file", _dl)

    target = cache / f"orekit_data_{SHA_NEW[:8]}.zip"
    assert orekit_utils.resolve_orekit_data() == target
    assert target.read_bytes() == b"PK new"
    assert calls == [
        (
            f"https://gitlab.orekit.org/orekit/orekit-data/-/archive/{SHA_NEW}/"
            f"orekit-data-{SHA_NEW}.zip",
            str(target) + ".part",
        )
    ]
    assert not old.exists()
    assert legacy.exists()  # DATA_DIR is never modified
    assert not Path(str(target) + ".part").exists()


def test_or04_failed_download_keeps_old_zip(monkeypatch, dirs):
    cache, _ = dirs
    old = _zip(cache / "orekit_data_11111111.zip", age_s=100)
    _sha_ok(monkeypatch)

    def _dl(url, filename):
        Path(filename).write_bytes(b"partial")
        raise requests.ConnectionError("dropped")

    monkeypatch.setattr(utils, "dl_file", _dl)

    with pytest.warns(RuntimeWarning, match="dropped"):
        assert orekit_utils.resolve_orekit_data() == old
    assert old.exists()
    assert list(cache.glob("*.part")) == []


# ---------------------------------------------------------------------------
# OR-05 helper jar is package data
# ---------------------------------------------------------------------------
def test_or05_helper_jar_in_installed_package():
    jar = orekit_utils.helper_jar_path()
    assert jar.is_file()
    assert jar.name == orekit_utils.JAR_NAME
    pkg = Path(contigo.__file__).resolve().parent
    assert jar.resolve().is_relative_to(pkg), (jar, pkg)


# ---------------------------------------------------------------------------
# PK-04 offline JVM start from a local zip (orekit tier, no network)
# ---------------------------------------------------------------------------
PK04_SCRIPT = """
import jpype
import contigo.config as config
from contigo.contigo_utils import orekit_utils
orekit_utils.start_orekit()
print("PK04", jpype.isJVMStarted(), config.state["orekit_loaded"], flush=True)
"""


def _local_zip_cache_root() -> tuple[Path, Path] | None:
    """``(zip, cache root to use)`` for an Orekit data zip already on disk.

    Looks in the developer's real per-user cache, the session cache and
    ``DATA_DIR`` (always searched by ``resolve_orekit_data``).
    """
    import platformdirs

    roots = [
        Path(platformdirs.user_cache_dir("contigo", appauthor=False)),
        config.cache_dir(create=False),
    ]
    for root in roots:
        zips = sorted((root / "orekit").glob("orekit_data_*.zip"))
        if zips:
            return zips[-1], root
    zips = sorted(Path(config.DATA_DIR).glob("orekit_data_*.zip"))
    if zips:
        return zips[-1], config.cache_dir(create=False)
    return None


@pytest.mark.orekit
def test_pk04_start_orekit_offline_uses_local_zip(tmp_path):
    found = _local_zip_cache_root()
    if found is None:
        pytest.skip("no local orekit_data_*.zip in the cache or DATA_DIR")
    _, cache_root = found

    pkg_root = str(Path(contigo.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (pkg_root, env.get("PYTHONPATH")) if p
    )
    env["CONTIGO_OFFLINE"] = "1"
    env["CONTIGO_CACHE_DIR"] = str(cache_root)
    proc = subprocess.run(
        [sys.executable, "-c", PK04_SCRIPT],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PK04")]
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert lines == ["PK04 True True"], proc.stdout[-2000:]
