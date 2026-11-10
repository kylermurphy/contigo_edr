"""Time-scale conversions built on SPICE (leap-second kernel handling).

added: 26/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import spiceypy as spice

import contigo.config as config
from contigo.contigo_utils import kernels

logger = logging.getLogger(__name__)


def spice_time(stime: npt.ArrayLike, tscale: str, out_scale: str):
    """Convert times between SPICE time scales.

    Parameters
    ----------
    stime : pandas.DatetimeIndex
        Times to convert. Interpreted on the scale given by ``tscale``.
    tscale : {"GPS", "TAI", "UTC", "ET", "TDB"}
        Time scale of ``stime``. Case-insensitive.
    out_scale : {"GPS", "TAI", "UTC", "ET", "TDB"}
        Time scale of the output. Case-insensitive.

    Returns
    -------
    numpy.ndarray
        Seconds past J2000 (2000-01-01 12:00:00) on ``out_scale``, as floats.
        For ``tscale == out_scale == "UTC"`` this is the calendar difference
        between ``stime`` and J2000 (leap seconds are not counted), the same
        convention used for the other scales when the scale is unchanged.

    Raises
    ------
    ValueError
        If ``tscale`` or ``out_scale`` is not one of the allowed scales, or if
        a non-UTC ``tscale`` is converted to ``out_scale == "UTC"`` (not yet
        implemented).
    """
    allowed = {"GPS", "TAI", "UTC", "ET", "TDB"}
    tscale = tscale.upper()
    out_scale = out_scale.upper()
    if tscale not in allowed:
        raise ValueError(f"tscale must be one of {allowed}")
    if out_scale not in allowed:
        raise ValueError(f"out_scale must be one of {allowed}")

    # make sure the leap second kernel is loaded
    # if it isn't check for it or download it and load it
    check_lpsk()

    if tscale == "UTC":
        # t_str = pd.to_datetime(np.array(stime)).strftime('%d %b %Y %H:%M:%S.%f')
        # et_t = np.array([spice.utc2et(sp_in) for sp_in in t_str])
        if out_scale == "UTC":
            # UTC -> UTC: no conversion, just seconds past J2000
            j2000 = pd.Timestamp("2000-01-01 12:00:00")
            return np.asarray((stime - j2000).total_seconds(), dtype=float)
        et_t = spice.datetime2et(stime)
        if out_scale not in {"ET", "TDB"}:
            out_t = np.array([spice.unitim(sp_in, "ET", out_scale) for sp_in in et_t])
        else:
            out_t = et_t
    elif out_scale == "UTC":
        raise ValueError("Converting to UTC not yet implemented")
    else:
        j2000 = pd.Timestamp("2000-01-01 12:00:00")
        spj2000 = ((stime - j2000).total_seconds()).to_list()
        out_t = np.array([spice.unitim(sp_in, tscale, out_scale) for sp_in in spj2000])

    return out_t


def check_lpsk():
    """Make sure the SPICE leap-second kernel is loaded.

    If no kernel named ``config.LEAP_FILE`` is in the SPICE kernel pool, it is
    looked up in the per-user cache (``config.cache_dir("spice")``), then in
    the bundled ``config.DATA_DIR``, and furnished. If it is in neither, it is
    first downloaded from the NAIF generic kernels server into the cache.
    There is no check for a newer version on NAIF (``SPICEEphem`` does that).

    Raises
    ------
    contigo.config.OfflineError
        If the kernel must be downloaded and ``CONTIGO_OFFLINE`` is set.
    Exception
        Any other error raised by the download (``utils.dl_file``) or by
        ``spiceypy.furnsh`` is propagated unchanged.
    """
    leaps_f = config.LEAP_FILE
    if kernels.is_loaded(leaps_f):
        return
    kernels.furnsh_kernel(kernels.ensure_kernel(leaps_f, check_remote=False))
