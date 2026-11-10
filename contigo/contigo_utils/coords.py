"""Coordinate conversions: ECEF <-> WGS-84 geodetic.

Positions are metres and angles radians. Pure NumPy, no SPICE or network.

added: 10/11/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from contigo.contigo_utils.constants import WGS84_A, WGS84_E2

logger = logging.getLogger(__name__)


def _as_xyz(arr: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return ``arr`` as float with a trailing axis of length 3, else raise."""
    out = np.asarray(arr, dtype=float)
    if out.ndim == 0 or out.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3); got {out.shape}.")
    return out


def ecef_to_geodetic(r_ecef: npt.ArrayLike, n_iter: int = 5) -> npt.NDArray[np.float64]:
    """Convert ECEF positions to WGS-84 geodetic coordinates.

    Parameters
    ----------
    r_ecef : array_like
        (..., 3) ECEF position [x, y, z] in metres.
    n_iter : int, optional
        Fixed-point iterations for latitude, by default 5.

    Returns
    -------
    numpy.ndarray
        (..., 3) [lat, lon, alt] with lat/lon in radians and alt in metres
        above the ellipsoid. lat is in [-pi/2, pi/2], lon in [-pi, pi].

    Raises
    ------
    ValueError
        If the last axis of ``r_ecef`` is not 3.
    """
    r = _as_xyz(r_ecef, "r_ecef")
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(n_iter):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + WGS84_E2 * n * np.sin(lat), p)
    # this form of the height is well behaved at the poles
    alt = (
        p * np.cos(lat)
        + z * np.sin(lat)
        - WGS84_A * np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    )
    return np.stack([lat, lon, alt], axis=-1)


def geodetic_to_ecef(llh: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert WGS-84 geodetic coordinates to ECEF positions.

    Parameters
    ----------
    llh : array_like
        (..., 3) [lat, lon, alt] with lat/lon in radians and alt in metres.

    Returns
    -------
    numpy.ndarray
        (..., 3) ECEF position [x, y, z] in metres.

    Raises
    ------
    ValueError
        If the last axis of ``llh`` is not 3.
    """
    a = _as_xyz(llh, "llh")
    lat, lon, alt = a[..., 0], a[..., 1], a[..., 2]
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)
