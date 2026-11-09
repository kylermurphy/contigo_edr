"""Tests for ``contigo.contigo_utils.utils`` network helpers (UT-01, PR T2).

``wf_mtime`` and ``dl_file`` are replaced by the autouse ``_no_network``
fixture, so the originals are captured when this module is imported (before
any fixture runs). ``requests`` is mocked inside ``contigo.contigo_utils.utils``;
nothing here touches the network.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

from datetime import datetime

import pytest
import requests
from dateutil import tz

from contigo.contigo_utils import utils
from contigo.contigo_utils.utils import dl_file as _real_dl_file
from contigo.contigo_utils.utils import wf_mtime as _real_wf_mtime

URL = "https://example.invalid/file.bin"


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code=200, headers=None, raw=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.raw = raw

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")


class FakeRaw:
    """Byte stream with the ``read(amt, decode_content=...)`` signature of urllib3."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, amt=-1, decode_content=False):
        if amt is None or amt < 0:
            amt = len(self._data) - self._pos
        chunk = self._data[self._pos : self._pos + amt]
        self._pos += len(chunk)
        return chunk


def test_guard_replaced_the_module_functions():
    """UT-01 guard: the autouse fixture replaced the originals, which we captured."""
    assert utils.wf_mtime is not _real_wf_mtime
    assert utils.dl_file is not _real_dl_file


# ---------------------------------------------------------------------------
# UT-01: wf_mtime
# ---------------------------------------------------------------------------
def test_ut01_wf_mtime_parses_last_modified(monkeypatch):
    """UT-01: ``Last-Modified`` is parsed to a GMT-aware datetime.

    Source: the RFC 7231 example date 'Wed, 21 Oct 2015 07:28:00 GMT'.
    """
    seen = []

    def fake_head(url, *args, **kwargs):
        seen.append(url)
        return FakeResponse(headers={"Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT"})

    monkeypatch.setattr(utils.requests, "head", fake_head)

    out = _real_wf_mtime(URL)

    assert seen == [URL]
    assert out == datetime(2015, 10, 21, 7, 28, tzinfo=tz.gettz("GMT"))
    assert out.utcoffset().total_seconds() == 0


def test_ut01_wf_mtime_missing_header_returns_none(monkeypatch):
    """UT-01: a response without ``Last-Modified`` gives ``None`` (docstring)."""
    monkeypatch.setattr(utils.requests, "head", lambda url, **kw: FakeResponse())
    assert _real_wf_mtime(URL) is None


def test_ut01_wf_mtime_request_exception_returns_none(monkeypatch):
    """UT-01: a ``RequestException`` is swallowed and gives ``None`` (docstring)."""

    def boom(url, **kw):
        raise requests.exceptions.ConnectionError("no route to host")

    monkeypatch.setattr(utils.requests, "head", boom)
    assert _real_wf_mtime(URL) is None


def test_ut01_wf_mtime_http_error_returns_none(monkeypatch):
    """UT-01: a 404 (``raise_for_status`` raises ``HTTPError``) gives ``None``."""
    monkeypatch.setattr(utils.requests, "head", lambda url, **kw: FakeResponse(404))
    assert _real_wf_mtime(URL) is None


# ---------------------------------------------------------------------------
# dl_file
# ---------------------------------------------------------------------------
def test_dl_file_writes_stream_to_disk(tmp_path, monkeypatch):
    """dl_file: the streamed bytes are written to the target, parents created.

    Expected: the file holds exactly the bytes served by the fake ``get``.
    """
    data = bytes(range(256)) * 40
    monkeypatch.setattr(
        utils.requests,
        "get",
        lambda url, **kw: FakeResponse(
            200, {"Content-Length": str(len(data))}, FakeRaw(data)
        ),
    )
    target = tmp_path / "sub" / "dir" / "file.bin"

    _real_dl_file(URL, str(target))

    assert target.read_bytes() == data


def test_dl_file_unknown_length_asks_head(tmp_path, monkeypatch):
    """dl_file: with no ``Content-Length`` the size is looked up with HEAD.

    Expected: one HEAD request, and the bytes are still written intact.
    """
    data = b"hello contigo" * 100
    heads = []
    monkeypatch.setattr(
        utils.requests, "get", lambda url, **kw: FakeResponse(200, {}, FakeRaw(data))
    )

    def fake_head(url, **kw):
        heads.append(url)
        return FakeResponse(200, {"Content-Length": str(len(data))})

    monkeypatch.setattr(utils.requests, "head", fake_head)
    target = tmp_path / "file.bin"

    _real_dl_file(URL, str(target))

    assert heads == [URL]
    assert target.read_bytes() == data


def test_dl_file_http_error_raises_and_writes_nothing(tmp_path, monkeypatch):
    """dl_file: a 4xx status raises ``HTTPError`` (via ``raise_for_status``)."""
    monkeypatch.setattr(utils.requests, "get", lambda url, **kw: FakeResponse(404))
    target = tmp_path / "file.bin"
    with pytest.raises(requests.exceptions.HTTPError):
        _real_dl_file(URL, str(target))
    assert not target.exists()


def test_dl_file_other_non_200_raises_runtime_error(tmp_path, monkeypatch):
    """dl_file: a non-200 status that ``raise_for_status`` accepts (3xx) raises.

    Source: the explicit ``RuntimeError`` after ``raise_for_status``.
    """
    monkeypatch.setattr(utils.requests, "get", lambda url, **kw: FakeResponse(304))
    target = tmp_path / "file.bin"
    with pytest.raises(RuntimeError, match="304"):
        _real_dl_file(URL, str(target))
    assert not target.exists()
