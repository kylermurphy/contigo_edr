"""Config and State values for CONTIGO

Data and network
----------------
- ``DATA_DIR`` / :func:`data_path`: bundled, read-only package data.
- :func:`cache_dir`: per-user writable cache for downloaded files
  (``$CONTIGO_CACHE_DIR`` if set, else the platformdirs user cache dir).
- :func:`is_offline`: ``CONTIGO_OFFLINE=1`` forbids every download; helpers
  raise :class:`OfflineError` instead of touching the network.

added: 19/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import platformdirs

# A library must not configure the root logger; applications decide (R4).
logging.getLogger("contigo").addHandler(logging.NullHandler())

_TRUTHY = {"1", "true", "yes", "on"}


def is_offline() -> bool:
    """True if ``CONTIGO_OFFLINE`` is set to 1/true/yes/on (any case).

    The environment is read on every call, so it can be toggled at runtime.
    """
    return os.environ.get("CONTIGO_OFFLINE", "").strip().lower() in _TRUTHY


class OfflineError(RuntimeError):
    """Raised when a download is needed but ``CONTIGO_OFFLINE`` is set."""


def cache_dir(sub: str = "") -> Path:
    """Per-user writable cache directory, created if missing.

    Resolved on every call: ``$CONTIGO_CACHE_DIR`` if set, else
    ``platformdirs.user_cache_dir("contigo")``. ``sub`` names a subdirectory
    (e.g. ``"spice"``, ``"gravity"``, ``"orekit"``).
    """
    root = os.environ.get("CONTIGO_CACHE_DIR") or platformdirs.user_cache_dir(
        "contigo", appauthor=False
    )
    d = Path(root).expanduser()
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_path() -> str:
    """Directory of the bundled, read-only package data (never created)."""
    try:
        from importlib.resources import files

        d = Path(str(files("contigo") / "data"))
    except (ImportError, TypeError):
        d = Path(__file__).parent / "data"
    return str(d.resolve())


def find_data_file(name: str | os.PathLike, sub: str = "") -> Path | None:
    """Locate a data file; return its path, or None if it is not found.

    Lookup order: (1) ``name`` as an existing path (absolute or relative),
    (2) ``cache_dir(sub) / basename``, (3) ``DATA_DIR / basename``.
    """
    p = Path(name).expanduser()
    if p.is_file():
        return p
    for d in (cache_dir(sub), Path(DATA_DIR)):
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
