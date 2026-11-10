"""NAIF generic SPICE kernels: URLs, local lookup, download and loading.

Shared by ``SPICEEphem`` and ``time_utils.check_lpsk``.

- :func:`naif_url`: NAIF ``generic_kernels`` URL of a kernel, from its type.
- :func:`ensure_kernel`: local path of a kernel. Looked up with
  ``config.find_data_file(name, "spice")`` (per-user cache first, then the
  bundled ``DATA_DIR``). A missing kernel, or (online, when asked) a NAIF copy
  newer than the local file, is downloaded into ``config.cache_dir("spice")``;
  ``DATA_DIR`` is never written to. A download goes to a ``.part`` file and
  replaces the cached kernel only when complete. Offline there is no
  freshness check and a missing kernel raises ``config.OfflineError``.
- :func:`furnsh_kernel`: load a kernel once. Loaded kernels are compared by
  ``os.path.normcase(os.path.realpath(p))``; a loaded kernel with the same
  file name at another path is unloaded first, so a stale copy and its
  replacement are never in the pool together (later kernels take priority).
- :func:`check_pck`: make Earth orientation data for given epochs available
  (``ITRF93``), updating the PCK once per session if needed, else falling
  back to the low-precision ``IAU_EARTH`` frame. Used by
  :mod:`contigo.contigo_utils.coords`.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import requests
import spiceypy as spice
from spiceypy.utils.support_types import SPICEDOUBLE_CELL

import contigo.config as config
from contigo.contigo_utils import utils

logger = logging.getLogger(__name__)

NAIF_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

#: Kernel type -> directory under ``NAIF_BASE_URL``
NAIF_DIRS = {"lsk": "lsk", "pck": "pck", "spk": "spk/planets"}

#: File extension -> kernel type
_KIND_BY_EXT = {".tls": "lsk", ".bpc": "pck", ".tpc": "pck", ".bsp": "spk"}

#: SPICE body ID that ``pckcov`` reports for the ITRF93 binary PCK
ITRF93_PCK_ID = 3000
#: Earth-fixed frame names returned by :func:`check_pck`
ITRF93 = "ITRF93"
IAU_EARTH = "IAU_EARTH"

#: Errors that mean "could not get a newer kernel"; callers log and carry on
NET_ERRORS = (config.OfflineError, requests.exceptions.RequestException, OSError)

#: The ``contigo`` package directory, used to point warnings at the user's code
_PKG_DIR = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _external_stacklevel() -> int:
    """``stacklevel`` for ``warnings.warn`` that names the first caller outside contigo.

    Call it from the function that calls ``warnings.warn``. However deep the
    call chain inside contigo (``check_pck`` from ``coords``, from a plot
    function, ...), the warning then shows the user's line, and Python's
    once-per-location filter works per user call site.
    """
    level = 1
    frame = sys._getframe(1)  # the function that will call warnings.warn
    while frame is not None:
        raw = frame.f_code.co_filename
        if raw.startswith("<"):  # <stdin>, <ipython-input-...>: user code
            break
        if not os.path.normcase(os.path.abspath(raw)).startswith(_PKG_DIR + os.sep):
            break
        frame = frame.f_back
        level += 1
    if frame is None:  # every frame is inside contigo: name the outermost one
        level -= 1
    return level


def kernel_kind(name: str) -> str:
    """Kernel type (``"lsk"``, ``"pck"`` or ``"spk"``) from the file extension."""
    ext = os.path.splitext(name)[1].lower()
    try:
        return _KIND_BY_EXT[ext]
    except KeyError:
        raise ValueError(
            f"Cannot tell the SPICE kernel type of {name!r}; "
            f"known extensions: {sorted(_KIND_BY_EXT)}"
        ) from None


def naif_url(name: str, kind: str | None = None) -> str:
    """NAIF generic-kernels URL of kernel ``name``.

    ``kind`` (``"lsk"``, ``"pck"`` or ``"spk"``) defaults to
    :func:`kernel_kind` of ``name``; ``spk`` kernels live in ``spk/planets``.
    """
    kind = kernel_kind(name) if kind is None else kind
    if kind not in NAIF_DIRS:
        raise ValueError(f"kind must be one of {sorted(NAIF_DIRS)}, got {kind!r}")
    return f"{NAIF_BASE_URL}/{NAIF_DIRS[kind]}/{name}"


def _norm(p: str | os.PathLike) -> str:
    """Path key used to compare kernels (resolves links, case and slashes)."""
    return os.path.normcase(os.path.realpath(os.fspath(p)))


def _basename(p: str | os.PathLike) -> str:
    return os.path.normcase(os.path.basename(os.fspath(p)))


def loaded_kernels() -> list[str]:
    """File names of all kernels in the SPICE pool, as SPICE stores them."""
    return [spice.kdata(i, "ALL")[0] for i in range(spice.ktotal("ALL"))]


def is_loaded(name: str) -> bool:
    """True if a kernel with file name ``name`` is loaded (any directory)."""
    want = _basename(name)
    return any(_basename(k) == want for k in loaded_kernels())


def _unload_path(path: str | os.PathLike) -> bool:
    """Unload every pool entry that is the file ``path``; True if any was."""
    want = _norm(path)
    found = False
    for k in loaded_kernels():
        if _norm(k) == want:
            logger.info("Unloading kernel - %s", k)
            spice.unload(k)
            found = True
    return found


def _mtime_utc(path: str | os.PathLike) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)


def _download(url: str, target: Path) -> Path:
    """Download ``url`` to ``target`` without ever leaving a partial file there.

    The kernel is written to a per-process ``.part`` file next to ``target``,
    its size is checked against the server's ``Content-Length`` (when known),
    and only then is it moved into place with ``os.replace``. If the download
    fails, ``target`` and any loaded copy of it are left as they were, so a
    cut-off transfer cannot leave a truncated kernel in the cache.

    Raises
    ------
    config.OfflineError
        If ``CONTIGO_OFFLINE`` is set.
    requests.exceptions.RequestException
        On a network or HTTP error.
    OSError
        If the download is shorter than the server said, or on a file error.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f"{target.name}.{os.getpid()}.part")
    logger.info("Downloading kernel %s to %s", url, target)
    try:
        expected = utils.dl_file(url, str(part))  # OfflineError when offline
        # 0 or None means the size is unknown (no Content-Length, or compressed)
        if expected:
            got = part.stat().st_size
            if got != expected:
                raise OSError(
                    f"Incomplete download of {url}: {got} of {expected} bytes"
                )
        # a binary kernel stays open while loaded (Windows cannot replace it)
        was_loaded = target.exists() and _unload_path(target)
        try:
            os.replace(part, target)
        except OSError:
            if was_loaded:  # keep the old kernel in the pool
                spice.furnsh(str(target))
            raise
    finally:
        with contextlib.suppress(OSError):
            part.unlink(missing_ok=True)
    return target


def ensure_kernel(name: str, check_remote: bool = True) -> Path:
    """Local path of NAIF kernel ``name``, downloading it if needed.

    Parameters
    ----------
    name : str
        Kernel file name, e.g. ``"naif0012.tls"``.
    check_remote : bool
        If True and not offline, download the NAIF copy when it is newer than
        the local file (``utils.wf_mtime``). If that download fails, the
        local file is returned (and the failure logged as a warning).

    Returns
    -------
    pathlib.Path
        The resolved local file: the cached or bundled copy, or the new
        download in ``config.cache_dir("spice")``.

    Raises
    ------
    config.OfflineError
        If the kernel is not available locally and ``CONTIGO_OFFLINE`` is set.
    """
    url = naif_url(name)
    target = config.cache_dir("spice", create=False) / name
    local = config.find_data_file(name, "spice")
    if local is None:
        return _download(url, target)
    if not check_remote:
        return local
    if config.is_offline():
        logger.info("CONTIGO_OFFLINE set; using local kernel %s", local)
        return local
    mod_url = utils.wf_mtime(url)
    if mod_url is None:
        logger.info("Could not determine modification time of %s", url)
    elif mod_url > _mtime_utc(local):
        logger.info("NAIF has a newer version of %s", name)
        try:
            return _download(url, target)
        except NET_ERRORS as exc:
            logger.warning(
                "Could not download the newer %s (%s); using %s", name, exc, local
            )
    return local


def furnsh_kernel(path: str | os.PathLike) -> bool:
    """Load the kernel file ``path`` unless it is already loaded.

    A loaded kernel with the same file name but a different path is unloaded
    first. Returns True if ``spice.furnsh`` was called.
    """
    path = os.fspath(path)
    want, base = _norm(path), _basename(path)
    loaded = loaded_kernels()
    if any(_norm(k) == want for k in loaded):
        logger.info("Kernel already loaded - %s", path)
        return False
    for k in loaded:
        if _basename(k) == base:
            logger.info("Unloading kernel %s (replaced by %s)", k, path)
            spice.unload(k)
    logger.info("Loading Kernel - %s", path)
    spice.furnsh(path)
    return True


# ---------------------------------------------------------------------------
# Earth orientation
# ---------------------------------------------------------------------------
def pck_coverage(body: int = ITRF93_PCK_ID) -> SPICEDOUBLE_CELL:
    """Union of the coverage (ET seconds) of all loaded binary PCKs for ``body``."""
    cover = SPICEDOUBLE_CELL(2000)
    for i in range(spice.ktotal("PCK")):
        spice.pckcov(spice.kdata(i, "PCK")[0], body, cover)
    return cover


def pck_covers(et_min: float, et_max: float) -> bool:
    """True if one interval of the loaded PCKs contains ``[et_min, et_max]``."""
    return bool(spice.wnincd(et_min, et_max, pck_coverage()))


def _coverage_text() -> str:
    cover = pck_coverage()
    n = spice.wncard(cover)
    if n == 0:
        return "no loaded PCK covers ITRF93"
    start = spice.et2utc(spice.wnfetd(cover, 0)[0], "ISOC", 0)
    end = spice.et2utc(spice.wnfetd(cover, n - 1)[1], "ISOC", 0)
    return f"the loaded PCK covers {start} to {end} UTC"


def check_pck(
    et: npt.ArrayLike | None = None,
    *,
    fallback: Literal["warn", "raise"] = "warn",
) -> str:
    """Make Earth orientation data for ``et`` available; return the frame name.

    Furnishes the leap-second and Earth PCK kernels (cache, then bundled; no
    network). If ``et`` is not covered, one attempt per session is made to
    download a newer PCK into ``config.cache_dir("spice")``; this is skipped
    if ``SPICEEphem`` already did its once-per-session NAIF check
    (``config.state["kernel_downloaded"]``) or offline. If ``et`` is still not
    covered, the low-precision ``IAU_EARTH`` frame is returned with a warning,
    or ``RuntimeError`` is raised when ``fallback='raise'``.

    ``IAU_EARTH`` ignores nutation, polar motion and UT1-UTC: expect position
    errors of order 10 km at LEO. That is fine for a plot, not for EDR.

    Parameters
    ----------
    et : array_like, optional
        SPICE ephemeris times (TDB seconds past J2000) that will be
        transformed. If None only the kernels are loaded and ``'ITRF93'`` is
        returned.
    fallback : {'warn', 'raise'}, optional
        Action when ``et`` is not covered, by default ``'warn'``.

    Returns
    -------
    str
        ``'ITRF93'`` or ``'IAU_EARTH'``: the Earth-fixed frame to use.

    Raises
    ------
    RuntimeError
        If ``et`` is not covered and ``fallback='raise'``.
    ValueError
        If ``fallback`` is not ``'warn'`` or ``'raise'``.
    """
    if fallback not in ("warn", "raise"):
        raise ValueError("fallback must be 'warn' or 'raise'")
    for name in (config.LEAP_FILE, config.PCK_FILE):
        furnsh_kernel(ensure_kernel(name, check_remote=False))
    if et is None:
        return ITRF93
    et = np.asarray(et, dtype=float)
    if et.size == 0:
        return ITRF93
    et_min, et_max = float(et.min()), float(et.max())
    if pck_covers(et_min, et_max):
        return ITRF93

    if not (config.state["kernel_downloaded"] or config.state["pck_refreshed"]):
        config.state["pck_refreshed"] = True  # one attempt per session, pass or fail
        try:
            furnsh_kernel(ensure_kernel(config.PCK_FILE))
        except NET_ERRORS as exc:
            logger.info("Could not update %s: %s", config.PCK_FILE, exc)
        if pck_covers(et_min, et_max):
            return ITRF93

    msg = (
        "No Earth orientation data (ITRF93) for the requested epochs; "
        f"{_coverage_text()}."
    )
    if fallback == "raise":
        raise RuntimeError(msg)
    warnings.warn(
        msg + " Using low-precision IAU_EARTH (position error of order 10 km).",
        RuntimeWarning,
        stacklevel=_external_stacklevel(),
    )
    furnsh_kernel(ensure_kernel(config.TPC_FILE, check_remote=False))
    return IAU_EARTH
