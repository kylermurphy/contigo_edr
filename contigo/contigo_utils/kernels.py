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

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import spiceypy as spice

import contigo.config as config
from contigo.contigo_utils import utils

logger = logging.getLogger(__name__)

NAIF_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

#: Kernel type -> directory under ``NAIF_BASE_URL``
NAIF_DIRS = {"lsk": "lsk", "pck": "pck", "spk": "spk/planets"}

#: File extension -> kernel type
_KIND_BY_EXT = {".tls": "lsk", ".bpc": "pck", ".tpc": "pck", ".bsp": "spk"}


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
