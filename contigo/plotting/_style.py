"""Shared plotting style: lazy matplotlib import, colours and rc settings.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable
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


N_COLORS = 20  # size of the tab20 colormap


def _tab20(idx: int) -> tuple[float, float, float, float]:
    """RGBA of entry ``idx`` of ``tab20`` without importing pyplot."""
    try:
        import matplotlib
    except ImportError as err:
        raise ImportError(
            "Plotting needs matplotlib: pip install 'contigo[plot]'"
        ) from err
    r, g, b, a = matplotlib.colormaps["tab20"](idx % N_COLORS)
    return (float(r), float(g), float(b), float(a))


def _bin(sc_id: Any) -> int:
    """Stable colour bin of an ID (``crc32``, not the per-process ``hash``)."""
    return zlib.crc32(str(sc_id).encode()) % N_COLORS


def color_for(sc_id: Any) -> tuple[float, float, float, float]:
    """Return the stable RGBA colour of a single spacecraft ID.

    The colour comes from ``zlib.crc32(str(sc_id))`` modulo 20 into the
    ``tab20`` colormap, so it is the same across runs and processes (unlike
    ``hash()``, which is randomised per process). Two IDs can share a bin;
    when drawing several spacecraft use :func:`color_map`, which resolves
    clashes.

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
    return _tab20(_bin(sc_id))


def color_map(sc_ids: Iterable[Any]) -> dict[Any, tuple[float, float, float, float]]:
    """Assign distinct, stable colours to the spacecraft drawn in one plot.

    Each ID starts at its :func:`color_for` bin. IDs are processed in sorted
    string order, and an ID whose bin is taken moves to the next free bin, so
    the result depends only on the *set* of IDs (not their order). An ID
    with no clash always gets its :func:`color_for` colour. Beyond 20 IDs,
    colours repeat.

    Parameters
    ----------
    sc_ids : iterable
        Spacecraft identifiers in the plot.

    Returns
    -------
    dict
        ``{sc_id: RGBA}`` keyed by the original IDs.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    ids = list(dict.fromkeys(sc_ids))
    taken: set[int] = set()
    out: dict[Any, tuple[float, float, float, float]] = {}
    for sc_id in sorted(ids, key=str):
        b = _bin(sc_id)
        if len(taken) < N_COLORS:
            while b in taken:
                b = (b + 1) % N_COLORS
        taken.add(b)
        out[sc_id] = _tab20(b)
    return {sc_id: out[sc_id] for sc_id in ids}


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
