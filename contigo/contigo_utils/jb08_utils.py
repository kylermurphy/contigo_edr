"""jb08_utils.py — Python wrapper for the JB2008DensityBatchHelper Java class.

Computes JB2008 atmospheric density at each point along a satellite track.

Two modes are supported, selected automatically based on whether space weather
indices are supplied:

* **Orekit-managed** (no indices passed): Orekit reads the SOLFSMY and DTCFILE
  space weather files from its bundled data archive. Simpler to call but limited
  to the date range covered by those files — typically ending a year or two
  before the Orekit data zip was published. Raises ``OrekitException`` outside
  that range.

* **User-supplied** (all nine indices passed): The caller provides arrays of
  F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, and DSTDTC interpolated to the
  satellite epochs. The Java helper linearly interpolates between supplied points,
  so daily space weather values pre-interpolated to the satellite time axis work
  directly. Use this for recent data, custom datasets, or dates outside the
  Orekit archive range.

Two coordinate systems are supported via the ``coords`` keyword:

* ``'ecef'`` (default): positions columns are [x, y, z] in metres.
* ``'geodetic'``: positions columns are [lat_deg, lon_deg, alt_m] where
  latitude and longitude are in degrees (WGS-84).

All nine space weather parameters are keyword-only and must either all be
supplied or all omitted — passing a partial set raises ``ValueError``.

added: 29/05/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

import contigo.config as config
if config.state['orekit_loaded'] is False:
    from contigo.contigo_utils import orekit_utils
    orekit_utils.start_orekit()

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.contigo.orekit_utils import JB2008DensityBatchHelper

# All nine JB2008 space weather parameter names in the order accepted by
# JB2008DensityBatchHelper.getDensity.
_SW_NAMES = ('f10', 'f10b', 's10', 's10b', 'xm10', 'xm10b', 'y10', 'y10b', 'dstdtc')

_VALID_COORDS = {'ecef', 'geodetic'}


def jb2008_density(
    positions: npt.NDArray[np.float64],
    times: Sequence[datetime],
    *,
    coords: Literal['ecef', 'geodetic'] = 'ecef',
    f10:    npt.NDArray[np.float64] | None = None,
    f10b:   npt.NDArray[np.float64] | None = None,
    s10:    npt.NDArray[np.float64] | None = None,
    s10b:   npt.NDArray[np.float64] | None = None,
    xm10:   npt.NDArray[np.float64] | None = None,
    xm10b:  npt.NDArray[np.float64] | None = None,
    y10:    npt.NDArray[np.float64] | None = None,
    y10b:   npt.NDArray[np.float64] | None = None,
    dstdtc: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute JB2008 atmospheric density along a satellite track.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Satellite state array. Column interpretation depends on ``coords``:

        * ``coords='ecef'`` (default): [x, y, z] in metres (ECEF Cartesian).
        * ``coords='geodetic'``: [lat_deg, lon_deg, alt_m] where latitude and
          longitude are in degrees (WGS-84) and altitude is in metres.

    times : sequence of datetime
        UTC datetimes corresponding to each row of ``positions``.
        Must be timezone-naive (i.e. ``tzinfo=None``).

    coords : {'ecef', 'geodetic'}, optional
        Coordinate system of ``positions``. Default is ``'ecef'``.

    Space weather indices (all keyword-only, all optional)
    ------------------------------------------------------
    All nine parameters must be supplied together or not at all.
    Each should be a 1-D array of length N with values interpolated to the
    satellite epochs. Typical sources: SOLFSMY and DTCFILE files from Space
    Environment Technologies (spacewx.com). When omitted, Orekit reads its
    bundled space weather archive automatically.

    f10 : ndarray, shape (N,)
        Solar flux index at 10.7 cm with 1-day lag (sfu).
    f10b : ndarray, shape (N,)
        81-day centred average of F10 (sfu).
    s10 : ndarray, shape (N,)
        Solar EUV proxy index S10.
    s10b : ndarray, shape (N,)
        81-day centred average of S10.
    xm10 : ndarray, shape (N,)
        Solar MUV proxy index XM10.
    xm10b : ndarray, shape (N,)
        81-day centred average of XM10.
    y10 : ndarray, shape (N,)
        Solar X-ray and Lyman-alpha proxy index Y10.
    y10b : ndarray, shape (N,)
        81-day centred average of Y10.
    dstdtc : ndarray, shape (N,)
        Exospheric temperature correction driven by geomagnetic activity (K).

    Returns
    -------
    ndarray, shape (N,)
        Atmospheric density in kg/m³ at each epoch.

    Raises
    ------
    ValueError
        If ``coords`` is not ``'ecef'`` or ``'geodetic'``, if only some space
        weather indices are supplied, or if ``positions`` and ``times`` have
        different lengths.
    OrekitException
        If Orekit-managed mode is used and the requested epochs fall outside
        the date range of the bundled space weather archive.

    Examples
    --------
    Orekit-managed, ECEF positions (simplest):

    >>> rho = jb2008_density(sc.state_ecef, sc.sc_utc)

    Orekit-managed, geodetic positions:

    >>> rho = jb2008_density(lla, times, coords='geodetic')

    User-supplied space weather, ECEF positions:

    >>> rho = jb2008_density(
    ...     sc.state_ecef, sc.sc_utc,
    ...     f10=f10, f10b=f10b, s10=s10, s10b=s10b,
    ...     xm10=xm10, xm10b=xm10b, y10=y10, y10b=y10b,
    ...     dstdtc=dstdtc,
    ... )
    """
    if coords not in _VALID_COORDS:
        raise ValueError(
            f"coords must be one of {_VALID_COORDS!r}, got {coords!r}"
        )

    sw_vals = (f10, f10b, s10, s10b, xm10, xm10b, y10, y10b, dstdtc)
    n_supplied = sum(v is not None for v in sw_vals)

    if 0 < n_supplied < len(_SW_NAMES):
        missing = [name for name, val in zip(_SW_NAMES, sw_vals) if val is None]
        raise ValueError(
            f"All nine space weather indices must be supplied together. "
            f"Missing: {missing}"
        )

    use_sw = n_supplied == len(_SW_NAMES)
    geodetic = coords == 'geodetic'

    positions = np.asarray(positions, dtype=np.float64)
    n = len(times)

    if positions.shape != (n, 3):
        raise ValueError(
            f"positions must have shape (N, 3), got {positions.shape}"
        )

    # Build Orekit AbsoluteDate from the first epoch, then express all
    # subsequent epochs as second offsets from it.
    utc = TimeScalesFactory.getUTC()
    t0 = times[0]
    ref_date = AbsoluteDate(
        t0.year, t0.month, t0.day,
        t0.hour, t0.minute, float(t0.second),
        utc,
    )
    offsets = np.array(
        [(t - t0).total_seconds() for t in times],
        dtype=np.float64,
    )

    if use_sw:
        density = JB2008DensityBatchHelper.getDensity(
            ref_date, offsets, positions,
            geodetic,
            np.asarray(f10,    dtype=np.float64),
            np.asarray(f10b,   dtype=np.float64),
            np.asarray(s10,    dtype=np.float64),
            np.asarray(s10b,   dtype=np.float64),
            np.asarray(xm10,   dtype=np.float64),
            np.asarray(xm10b,  dtype=np.float64),
            np.asarray(y10,    dtype=np.float64),
            np.asarray(y10b,   dtype=np.float64),
            np.asarray(dstdtc, dtype=np.float64),
        )
    else:
        density = JB2008DensityBatchHelper.getDensityFromOrekitData(
            ref_date, offsets, positions,
            geodetic,
        )

    return np.array(density, dtype=np.float64)
