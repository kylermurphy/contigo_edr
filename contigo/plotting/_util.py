"""Pure helpers shared by the plotting functions (no matplotlib needed).

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from contigo.constellation import Constellation
from contigo.spacecraft import Spacecraft


def as_spacecraft_dict(
    data: Spacecraft | Constellation, ids: Sequence[Any] | None = None
) -> dict:
    """Normalise plot input to a ``{spacecraft_id: Spacecraft}`` dict.

    Parameters
    ----------
    data : Spacecraft or Constellation
        A ``Spacecraft`` is split with ``Spacecraft.split_by_id()``; a
        ``Constellation`` contributes its ``.spacecraft`` dict.
    ids : iterable or scalar, optional
        Subset of spacecraft IDs to keep. IDs are compared as strings
        (``str(key) == str(requested)``, bytes decoded) because they are often
        pandas categories or numpy strings. A single scalar ID is accepted. The
        original keys and the order of ``data`` are kept. By default all
        spacecraft are returned.

    Returns
    -------
    dict
        ``{spacecraft_id: Spacecraft}`` using the original keys.

    Raises
    ------
    TypeError
        If ``data`` is neither a ``Spacecraft`` nor a ``Constellation``.
    ValueError
        If a requested ID is not present; the message lists the available IDs.
    """
    if isinstance(data, Spacecraft):
        sc_dict = data.split_by_id()
    elif isinstance(data, Constellation):
        sc_dict = dict(data.spacecraft)
    else:
        raise TypeError(
            f"data must be a Spacecraft or Constellation, got {type(data).__name__}"
        )

    if ids is None:
        return sc_dict

    if isinstance(ids, (str, bytes)) or not isinstance(ids, Iterable):
        ids = [ids]  # a single ID
    requested = [_id_str(i) for i in ids]
    available = [_id_str(k) for k in sc_dict]
    missing = [r for r in requested if r not in available]
    if missing:
        raise ValueError(
            f"Unknown spacecraft ID(s) {missing}; available IDs: {available}"
        )
    wanted = set(requested)
    return {k: v for k, v in sc_dict.items() if _id_str(k) in wanted}


def _id_str(sc_id: Any) -> str:
    """Comparable string form of an ID (bytes from HDF files are decoded)."""
    if isinstance(sc_id, (bytes, np.bytes_)):
        return sc_id.decode()
    return str(sc_id)


def decimate(n: int, max_points: int) -> npt.NDArray[np.int_]:
    """Choose at most ``max_points`` evenly spaced indices from ``n`` samples.

    Parameters
    ----------
    n : int
        Number of samples.
    max_points : int
        Maximum number of indices to return; must be at least 2.

    Returns
    -------
    numpy.ndarray
        ``arange(n)`` if ``n <= max_points``; otherwise sorted unique indices
        that include ``0`` and ``n - 1``.

    Raises
    ------
    ValueError
        If ``max_points < 2`` or ``n < 0``.
    """
    if max_points < 2:
        raise ValueError(f"max_points must be >= 2, got {max_points}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def split_antimeridian(
    lon: npt.ArrayLike, *arrays: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], ...]:
    """Break a track where longitude wraps so lines do not cross the map.

    A ``NaN`` is inserted after sample ``i`` wherever
    ``abs(lon[i + 1] - lon[i]) > pi``, in ``lon`` and in every companion array.

    Parameters
    ----------
    lon : array_like
        (N,) longitude in radians.
    *arrays : array_like
        Companion (N,) arrays (for example latitude) split at the same places.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lon_out, *arrays_out)`` as float arrays of length ``N + n_breaks``.

    Raises
    ------
    ValueError
        If ``lon`` is not 1-D or a companion array does not have the same
        length as ``lon``.
    """
    lon_a = np.asarray(lon, dtype=float)
    if lon_a.ndim != 1:
        raise ValueError(f"lon must be 1-D, got shape {lon_a.shape}")
    comp = [np.asarray(a, dtype=float) for a in arrays]
    for k, a in enumerate(comp):
        if a.shape != lon_a.shape:
            raise ValueError(
                f"companion array {k} has shape {a.shape}, expected {lon_a.shape}"
            )
    breaks = np.flatnonzero(np.abs(np.diff(lon_a)) > np.pi) + 1
    return tuple(np.insert(a, breaks, np.nan) for a in (lon_a, *comp))
