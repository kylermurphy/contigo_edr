"""Tests for the on-demand ICGEM gravity-field download (R2).

Fast tier: ``utils.dl_file`` is replaced by a fake that writes a small,
valid synthetic ``.gfc`` file and the registry by a fake entry for it.
Network tier: real ICGEM requests (EIGEN-2 download, EGM2008/EIGEN-6C4 sizes).

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import requests

import contigo.config as config
import contigo.forces.grav_utils as gu
from contigo.contigo_utils import utils
from contigo.forces.grav_pot import GravPot

FAKE = "FAKE.gfc"
FAKE_URL = "https://example.invalid/getmodel/gfc/FAKE.gfc"

GFC_TEXT = (
    "generating_institute test\n"
    "product_type gravity_field\n"
    "modelname FAKE\n"
    "earth_gravity_constant 0.3986004415E+15\n"
    "radius 0.6378136460E+07\n"
    "max_degree 2\n"
    "errors no\n"
    "key L M C S\n"
    "end_of_head ====\n"
    "gfc 0 0 0.100000000000e+01 0.000000000000e+00\n"
    "gfc 1 0 0.000000000000e+00 0.000000000000e+00\n"
    "gfc 1 1 0.000000000000e+00 0.000000000000e+00\n"
    "gfc 2 0 -.484165815935e-03 0.000000000000e+00\n"
    "gfc 2 1 0.000000000000e+00 0.000000000000e+00\n"
    "gfc 2 2 0.243930491835e-05 -.140027520233e-05\n"
)
GFC_BYTES = GFC_TEXT.encode("ascii")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Per-test CONTIGO_CACHE_DIR; return the gravity subdirectory path."""
    root = tmp_path / "cache"
    monkeypatch.setenv("CONTIGO_CACHE_DIR", str(root))
    monkeypatch.delenv("CONTIGO_OFFLINE", raising=False)
    monkeypatch.chdir(tmp_path)  # find_data_file also looks in the cwd
    return root / "gravity"


@pytest.fixture
def fake_registry(monkeypatch):
    """Replace the registry by a single fake entry matching GFC_BYTES."""
    model = gu.IcgemModel(
        url=FAKE_URL,
        size=len(GFC_BYTES),
        sha256=hashlib.sha256(GFC_BYTES).hexdigest(),
        lmax=2,
    )
    monkeypatch.setattr(gu, "ICGEM_MODELS", {FAKE: model})
    return model


@pytest.fixture
def fake_dl(monkeypatch):
    """Fake ``utils.dl_file``; returns the list of (url, filename) calls."""
    calls: list[tuple[str, str]] = []
    payload = {"data": GFC_BYTES}

    def _dl(url, filename):
        calls.append((url, filename))
        Path(filename).write_bytes(payload["data"])

    monkeypatch.setattr(utils, "dl_file", _dl)
    _dl.payload = payload
    _dl.calls = calls
    return _dl


def _leftovers(d: Path) -> list[Path]:
    return list(d.glob("*.part")) if d.exists() else []


def test_fetch_downloads_into_cache(cache, fake_registry, fake_dl):
    path = gu.fetch_icgem(FAKE)
    assert path == cache / FAKE
    assert path.read_bytes() == GFC_BYTES
    assert _leftovers(cache) == []
    assert len(fake_dl.calls) == 1
    url, part = fake_dl.calls[0]
    assert url == FAKE_URL
    assert Path(part).name.endswith(".part")


def test_size_mismatch_raises(cache, fake_registry, fake_dl, monkeypatch):
    monkeypatch.setitem(
        gu.ICGEM_MODELS, FAKE, fake_registry._replace(size=fake_registry.size + 1)
    )
    with pytest.raises(IOError, match="bytes, expected"):
        gu.fetch_icgem(FAKE)
    assert not (cache / FAKE).exists()
    assert _leftovers(cache) == []


def test_sha_mismatch_raises(cache, fake_registry, fake_dl, monkeypatch):
    monkeypatch.setitem(gu.ICGEM_MODELS, FAKE, fake_registry._replace(sha256="0" * 64))
    with pytest.raises(IOError, match="sha256"):
        gu.fetch_icgem(FAKE)
    assert not (cache / FAKE).exists()
    assert _leftovers(cache) == []


def test_header_mismatch_raises(cache, fake_registry, fake_dl, monkeypatch):
    monkeypatch.setitem(
        gu.ICGEM_MODELS, FAKE, fake_registry._replace(size=None, sha256=None, lmax=3)
    )
    with pytest.raises(IOError, match="max_degree"):
        gu.fetch_icgem(FAKE)
    assert not (cache / FAKE).exists()
    assert _leftovers(cache) == []


def test_missing_modelname_raises(cache, fake_registry, fake_dl, monkeypatch):
    monkeypatch.setitem(
        gu.ICGEM_MODELS, FAKE, fake_registry._replace(size=None, sha256=None)
    )
    fake_dl.payload["data"] = GFC_BYTES.replace(b"modelname FAKE\n", b"")
    with pytest.raises(IOError, match="modelname"):
        gu.fetch_icgem(FAKE)
    assert not (cache / FAKE).exists()
    assert _leftovers(cache) == []


def test_offline_raises_without_download(cache, fake_registry, fake_dl, monkeypatch):
    monkeypatch.setenv("CONTIGO_OFFLINE", "1")
    with pytest.raises(gu.PotentialFileNotFound) as exc:
        gu.resolve_pot_file(FAKE)
    msg = str(exc.value)
    assert FAKE_URL in msg
    assert str(cache) in msg
    assert "unset CONTIGO_OFFLINE" in msg
    assert isinstance(exc.value, ValueError)
    assert isinstance(exc.value, FileNotFoundError)
    assert fake_dl.calls == []


def test_unknown_name_raises_without_download(cache, fake_registry, fake_dl):
    with pytest.raises(gu.PotentialFileNotFound, match="cannot be found; searched"):
        gu.resolve_pot_file("NOPE.gfc")
    assert fake_dl.calls == []


def test_cache_hit_does_not_download(cache, fake_registry, fake_dl):
    cache.mkdir(parents=True)
    (cache / FAKE).write_bytes(GFC_BYTES)
    assert gu.resolve_pot_file(FAKE) == cache / FAKE
    assert fake_dl.calls == []


def test_missing_explicit_path_does_not_download(
    cache, fake_registry, fake_dl, tmp_path
):
    with pytest.raises(gu.PotentialFileNotFound, match="cannot be found"):
        gu.resolve_pot_file(str(tmp_path / "elsewhere" / FAKE))
    with pytest.raises(gu.PotentialFileNotFound):
        gu.resolve_pot_file(f"sub/{FAKE}")
    assert fake_dl.calls == []


def test_bundled_eigen2_resolves_without_download(cache, fake_dl):
    assert gu.resolve_pot_file("EIGEN-2.gfc") == Path(config.DATA_DIR) / "EIGEN-2.gfc"
    assert fake_dl.calls == []


def test_gravpot_end_to_end(cache, fake_registry, fake_dl, monkeypatch):
    monkeypatch.setitem(config.state, "pot_coef_loaded", False)
    monkeypatch.setitem(config.state, "pot_file", None)
    gp = GravPot(
        r=np.array([6_771_000.0, 7_000_000.0]),
        lat=np.radians([0.0, 45.0]),
        lon=np.radians([10.0, -100.0]),
        pot_file=FAKE,
        lmax=50,
    )
    assert Path(gp.pot_file) == cache / FAKE
    assert gp.lmax == 2
    gp.calc_pot()
    u = gp.get_pot()
    assert u.shape == (2,)
    assert np.all(np.isfinite(u)) and np.all(u > 0)
    assert len(fake_dl.calls) == 1
    # second use: cached file, no new download
    GravPot(pot_file=FAKE)
    assert len(fake_dl.calls) == 1


# ---------------------------------------------------------------------------
# Network tier
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_network_eigen2_download(tmp_path):
    path = gu.fetch_icgem("EIGEN-2.gfc", dest_dir=tmp_path)
    assert path == tmp_path / "EIGEN-2.gfc"
    assert _leftovers(tmp_path) == []
    _, _, meta = gu.read_icgem_coeff(str(path))
    assert meta["lmax"] == 140


@pytest.mark.network
@pytest.mark.parametrize("name", ["EGM2008.gfc", "EIGEN-6C4.gfc"])
def test_network_large_model_size(name):
    """The registry size equals the server's Content-Length (HEAD, no body).

    ICGEM serves the files with LF line endings; a copy checked out with CRLF
    is one byte per line larger and has a different sha256.
    """
    model = gu.ICGEM_MODELS[name]
    r = requests.head(
        model.url,
        allow_redirects=True,
        headers={"Accept-Encoding": "identity"},
        timeout=60,
    )
    assert r.status_code == 200
    assert int(r.headers["Content-Length"]) == model.size


@pytest.mark.network
@pytest.mark.slow
def test_network_egm2008_full_download(tmp_path):
    """Full ~250 MB download; fetch_icgem verifies size, sha256 and header."""
    path = gu.fetch_icgem("EGM2008.gfc", dest_dir=tmp_path)
    assert path.stat().st_size == gu.ICGEM_MODELS["EGM2008.gfc"].size
