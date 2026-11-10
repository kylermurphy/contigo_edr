"""NAIF generic SPICE kernels: URLs, local lookup, download and loading.

Shared by ``SPICEEphem`` and ``time_utils.check_lpsk``.

- :func:`naif_url`: NAIF ``generic_kernels`` URL of a kernel, from its type.
- :func:`ensure_kernel`: local path of a kernel. Looked up with
  ``config.find_data_file(name, "spice")`` (per-user cache first, then the
  bundled ``DATA_DIR``). A missing kernel, or (online, when asked) a NAIF copy
  newer than the local file, is downloaded into ``config.cache_dir("spice")``;
  ``DATA_DIR`` is never written to. Offline there is no freshness check and a
  missing kernel raises ``config.OfflineError``.
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

import logging
import os
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


def _unload_path(path: str | os.PathLike) -> None:
    """Unload every pool entry that is the file ``path``."""
    want = _norm(path)
    for k in loaded_kernels():
        if _norm(k) == want:
            logger.info("Unloading kernel - %s", k)
            spice.unload(k)


def _mtime_utc(path: str | os.PathLike) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)


def _download(url: str, target: Path) -> Path:
    # a binary kernel stays open while loaded: release it before overwriting
    if target.exists():
        _unload_path(target)
    logger.info("Downloading kernel %s to %s", url, target)
    utils.dl_file(url, str(target))  # raises config.OfflineError when offline
    return target


def ensure_kernel(name: str, check_remote: bool = True) -> Path:
    """Local path of NAIF kernel ``name``, downloading it if needed.

    Parameters
    ----------
    name : str
        Kernel file name, e.g. ``"naif0012.tls"``.
    check_remote : bool
        If True and not offline, download the NAIF copy when it is newer than
        the local file (``utils.wf_mtime``).

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
        return _download(url, target)
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
        stacklevel=3,
    )
    furnsh_kernel(ensure_kernel(config.TPC_FILE, check_remote=False))
    return IAU_EARTH
