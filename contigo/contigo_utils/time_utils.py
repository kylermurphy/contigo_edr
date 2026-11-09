"""Time-scale conversions built on SPICE (leap-second kernel handling).

added: 26/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

import logging
import os
import posixpath
import urllib.parse

import numpy as np
import numpy.typing as npt
import pandas as pd
import spiceypy as spice

import contigo.config as config
from contigo.contigo_utils import utils

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

    If the kernel (``config.LEAP_FILE``) is not in the SPICE kernel pool it is
    furnished from ``config.DATA_DIR``; if it is not there either, it is first
    downloaded from the NAIF generic kernels server.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    Exception
        Any error raised by the download (``utils.dl_file``) or by
        ``spiceypy.furnsh`` is propagated unchanged.
    """
    # get the leapsecond kernel and make sure it's
    # loaded, download it if we don't have it
    leaps_f = config.LEAP_FILE
    lp_kernel = os.path.join(config.DATA_DIR, leaps_f)

    sp_kcnt = spice.ktotal("ALL")
    sp_loaded = [os.path.basename(spice.kdata(i, "ALL")[0]) for i in range(sp_kcnt)]
    if os.path.exists(lp_kernel) and leaps_f not in sp_loaded:
        spice.furnsh(lp_kernel)  # need to check if kernels are loaded
    elif leaps_f not in sp_loaded:
        base_url = "https://naif.jpl.nasa.gov"
        base_pth = "/pub/naif/generic_kernels"
        leaps_d = "lsk"

        leaps_url = urllib.parse.urljoin(
            base_url, posixpath.join(base_pth, leaps_d, leaps_f)
        )

        logger.info("Downloading kernel - %s", lp_kernel)
        utils.dl_file(leaps_url, lp_kernel)
        spice.furnsh(lp_kernel)
