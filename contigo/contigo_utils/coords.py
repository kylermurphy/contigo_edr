"""Coordinate conversions: ECEF <-> WGS-84 geodetic.

Positions are metres and angles radians. Pure NumPy, no SPICE or network.

added: 10/11/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
import spiceypy as spice

from contigo.contigo_utils import kernels
from contigo.contigo_utils.constants import WGS84_A, WGS84_E2

logger = logging.getLogger(__name__)

_INERTIAL_FRAME = "J2000"


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


def _transform(
    state: npt.ArrayLike,
    et: npt.ArrayLike,
    to_inertial: bool,
    fallback: Literal["warn", "raise"],
) -> npt.NDArray[np.float64]:
    """Rotate (N, 3) positions or (N, 6) states between ECEF and J2000."""
    state = np.asarray(state, dtype=float)
    et = np.atleast_1d(np.asarray(et, dtype=float))
    if state.ndim != 2 or state.shape[1] not in (3, 6):
        raise ValueError(f"state must have shape (N, 3) or (N, 6); got {state.shape}.")
    if et.shape != (state.shape[0],):
        raise ValueError(f"et must have shape ({state.shape[0]},); got {et.shape}.")
    if not np.all(np.isfinite(et)):
        raise ValueError("et contains NaN or inf.")

    frame = kernels.check_pck(et, fallback=fallback)
    src, dst = (frame, _INERTIAL_FRAME) if to_inertial else (_INERTIAL_FRAME, frame)
    # sxform includes the Earth-rotation term: v_i = R (v_ecef + omega x r_ecef).
    # It is a Python loop (~5 us per epoch), so decimate before calling.
    func = spice.sxform if state.shape[1] == 6 else spice.pxform
    mats = np.array([func(src, dst, t) for t in et])
    return np.einsum("nij,nj->ni", mats, state)


def ecef_to_inertial(
    state_ecef: npt.ArrayLike,
    et: npt.ArrayLike,
    *,
    fallback: Literal["warn", "raise"] = "warn",
) -> npt.NDArray[np.float64]:
    """Transform Earth-fixed states to the inertial J2000 frame.

    Uses ``sxform('ITRF93', 'J2000')``. If the loaded Earth PCK does not cover
    ``et`` the frame falls back to the low-precision ``IAU_EARTH`` (position
    error of order 10 km, see :func:`contigo.contigo_utils.kernels.check_pck`).

    Parameters
    ----------
    state_ecef : array_like
        (N, 6) [x, y, z, vx, vy, vz] in m and m/s, with the velocity measured
        in the rotating frame (as ``Spacecraft.state_ecef``). An (N, 3) array
        of positions is also accepted and returns positions only.
    et : array_like
        (N,) SPICE ephemeris time (TDB seconds past J2000), as
        ``Spacecraft.sspice_et``.
    fallback : {'warn', 'raise'}, optional
        What to do when no high-precision orientation data covers ``et``:
        warn and use ``IAU_EARTH`` (default) or raise ``RuntimeError``.

    Returns
    -------
    numpy.ndarray
        Same shape as ``state_ecef``, in J2000.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or ``et`` is not finite.
    RuntimeError
        If ``fallback='raise'`` and ``et`` is not covered.
    """
    return _transform(state_ecef, et, True, fallback)


def inertial_to_ecef(
    state_inertial: npt.ArrayLike,
    et: npt.ArrayLike,
    *,
    fallback: Literal["warn", "raise"] = "warn",
) -> npt.NDArray[np.float64]:
    """Transform inertial J2000 states to Earth-fixed.

    Inverse of :func:`ecef_to_inertial`; same parameters, shapes and errors.
    """
    return _transform(state_inertial, et, False, fallback)
