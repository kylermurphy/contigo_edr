"""Config and State values for CONTIGO

Data and network
----------------
- ``DATA_DIR`` / :func:`data_path`: bundled, read-only package data.
- :func:`cache_dir`: per-user writable cache for downloaded files
  (``$CONTIGO_CACHE_DIR`` if set, else the platformdirs user cache dir).
- :func:`is_offline`: ``CONTIGO_OFFLINE=1`` blocks ``utils.dl_file`` (raises
  :class:`OfflineError`), ``utils.wf_mtime`` (returns None) and the Orekit
  GitLab SHA query; Orekit then starts from the newest local data zip
  (``contigo.contigo_utils.orekit_utils.resolve_orekit_data``).

added: 19/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

# A library must not configure the root logger; applications decide (R4).
# No NullHandler either: Python's lastResort handler then still prints
# WARNING and above to stderr when the application configures nothing.

_TRUTHY = {"1", "true", "yes", "on"}


def is_offline() -> bool:
    """True if ``CONTIGO_OFFLINE`` is set to 1/true/yes/on (any case).

    The environment is read on every call, so it can be toggled at runtime.
    """
    return os.environ.get("CONTIGO_OFFLINE", "").strip().lower() in _TRUTHY


class OfflineError(RuntimeError):
    """Raised when a download is needed but ``CONTIGO_OFFLINE`` is set."""


def cache_dir(sub: str = "", create: bool = True) -> Path:
    """Per-user writable cache directory.

    Resolved on every call: ``$CONTIGO_CACHE_DIR`` if set, else
    ``platformdirs.user_cache_dir("contigo")``. ``sub`` names a subdirectory
    (e.g. ``"spice"``, ``"gravity"``, ``"orekit"``). With ``create`` (the
    default) the directory is created if missing.
    """
    root = os.environ.get("CONTIGO_CACHE_DIR") or platformdirs.user_cache_dir(
        "contigo", appauthor=False
    )
    d = Path(root).expanduser()
    if sub:
        d = d / sub
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _files(pkg: str):
    from importlib.resources import files

    return files(pkg)


def data_path() -> str:
    """Directory of the bundled, read-only package data (never created)."""
    try:
        d = Path(str(_files("contigo") / "data"))
        if not d.is_dir():
            raise NotADirectoryError(d)
    except Exception:
        d = Path(__file__).parent / "data"
    return str(d.resolve())


def find_data_file(name: str | os.PathLike, sub: str = "") -> Path | None:
    """Locate a data file; return its resolved path, or None if not found.

    A path with a directory component (or an absolute path) is used as given:
    if it does not exist the result is None. A bare filename is looked up in
    the current directory, then ``cache_dir(sub)``, then ``DATA_DIR``.
    Nothing is created.
    """
    p = Path(name).expanduser()
    if p.is_file():
        return p.resolve()
    if p.is_absolute() or len(Path(name).parts) != 1:
        return None
    for d in (cache_dir(sub, create=False), Path(DATA_DIR)):
        cand = d / p.name
        if cand.is_file():
            return cand
    return None


# Static configuration values (constants)
DATA_DIR = data_path()
LEAP_FILE = "naif0012.tls"
PCK_FILE = "earth_latest_high_prec.bpc"

# Mutable State Values
state = {
    "orekit_loaded": False,
    "gmat_loaded": False,
    "gmatpy": None,
    "kernel_downloaded": False,
    "pot_coef_loaded": False,
    # Resolved full path (os.path.realpath) of the cached potential file
    "pot_file": None,
    "pot_lmax": None,
    "pot_clm": None,
    "pot_slm": None,
    "pot_r0": None,
    "pot_GM": None,
}
