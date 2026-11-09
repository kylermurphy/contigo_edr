"""Shared plotting style: lazy matplotlib import, colours and rc settings.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import zlib
from typing import Any

_RC: dict[str, Any] = {
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def require_matplotlib() -> Any:
    """Import and return ``matplotlib.pyplot``.

    Returns
    -------
    module
        The ``matplotlib.pyplot`` module.

    Raises
    ------
    ImportError
        If matplotlib is not installed; the message gives the install command.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "Plotting needs matplotlib: pip install 'contigo[plot]'"
        ) from err
    return plt


def color_for(sc_id: Any) -> tuple[float, float, float, float]:
    """Return a stable RGBA colour for a spacecraft ID.

    The colour comes from ``zlib.crc32(str(sc_id))`` modulo 20 into the
    ``tab20`` colormap, so it is the same across runs and processes (unlike
    ``hash()``, which is randomised per process).

    Parameters
    ----------
    sc_id : object
        Spacecraft identifier; only ``str(sc_id)`` is used.

    Returns
    -------
    tuple of float
        RGBA colour.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    plt = require_matplotlib()
    idx = zlib.crc32(str(sc_id).encode()) % 20
    r, g, b, a = plt.get_cmap("tab20")(idx)
    return (float(r), float(g), float(b), float(a))


def rc_context() -> Any:
    """Return a context manager applying the shared CONTIGO plot style.

    Use as ``with rc_context(): ...``; rcParams are restored on exit.

    Returns
    -------
    contextlib.AbstractContextManager
        ``matplotlib.pyplot.rc_context`` with the shared style.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    plt = require_matplotlib()
    return plt.rc_context(dict(_RC))
