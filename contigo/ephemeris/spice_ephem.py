"""Ephemeris provider using SPICE and SPICEYPY.

added: 04/03/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import spiceypy as spice

import contigo.config as config
from contigo.contigo_utils import kernels
from contigo.ephemeris.base import EphemerisProvider

logger = logging.getLogger(__name__)


class SPICEEphem(EphemerisProvider):
    """
    SPICE-backed ephemeris provider with unique-time optimization.
    """

    def __init__(
        self, ephemeris: str = "de440s", frame: str = "ITRF93", observer: str = "EARTH"
    ):

        self.frame = frame
        self.observer = observer
        self.ephemeris = ephemeris

        self.load_kernels()

    def __call__(
        self,
        body: npt.NDArray[np.str_] | list[str],
        utc_time: None = None,
        gps_time: None = None,
        ephem_time: np.ndarray | None = None,
    ):
        """
        Compute positions of celestial bodies.

        Parameters
        ----------
        ephem_time : npt.NDArray[np.float64]
            Ephemeris time array (ET) in seconds.
        gps_time : npt.NDArray[np.float64]
            GPS time array corresponding to ephem_time.
        utc_time : npt.NDArray[np.datetime64]
            UTC datetime array corresponding to ephem_time.


        Returns
        -------
        np.ndarray
            Array of shape (n_bodies, N, 3) containing position vectors.

        Notes
        -----
        This provider only uses ephem_time for its internal logic, but the interface
        allows for flexibility in case future providers need to use different
        time systems.
        """
        # find the unique values of et and return their indecies
        # return the inverse indices of that allow the reconstruction
        # of the orignal array
        unique_et, inv = np.unique(ephem_time, return_inverse=True)

        r_unique = (
            np.array(
                [
                    spice.spkpos(
                        bd.upper(), unique_et, self.frame, "NONE", self.observer
                    )[0]
                    for bd in body
                ]
            )
            * 1000.0
        )  # SPICE returns km; convert to m

        r_body = r_unique[:, inv, :]
        return r_body

    def load_kernels(self):
        """Find, download if needed, and load the SPICE kernels.

        Loads the ephemeris (``<ephemeris>.bsp``), leap-second
        (``config.LEAP_FILE``) and Earth orientation (``config.PCK_FILE``)
        kernels. Each is looked up in the per-user cache
        (``config.cache_dir("spice")``), then in the bundled ``DATA_DIR``;
        downloads always go to the cache. The first call in a session
        (``config.state["kernel_downloaded"]`` False) also checks NAIF for
        newer versions (:meth:`dl_kernels`); later calls only download
        kernels that are missing.

        A kernel already loaded from the same file (compared by normalised
        real path) is not loaded again. A loaded kernel with the same file name
        at another path (e.g. the bundled copy, after a newer one was
        downloaded) is unloaded first.
        """
        files = (f"{self.ephemeris}.bsp", config.LEAP_FILE, config.PCK_FILE)

        # check NAIF for newer kernels once per session
        if config.state["kernel_downloaded"] is False:
            sp_kernels = self.dl_kernels(*files)
        else:
            sp_kernels = [kernels.ensure_kernel(f, check_remote=False) for f in files]

        for fp in sp_kernels:
            kernels.furnsh_kernel(fp)

    def dl_kernels(self, ephem_f: str, leaps_f: str, pck_f: str) -> list[Path]:
        """Find or download the JPL SPICE kernels.

        Download ephemeris, leap second, and Earth orientation kernels from
        JPL's Navigation and Ancillary Information Facility (NAIF) into
        ``config.cache_dir("spice")`` if they are not found locally (cache,
        then bundled ``DATA_DIR``), or if the NAIF copy is newer than the local
        file. With ``CONTIGO_OFFLINE`` set the modification-time check is
        skipped and a missing kernel raises ``config.OfflineError``. Sets
        ``config.state["kernel_downloaded"]`` when all three are available.

        Parameters
        ----------
        ephem_f :str
            JPL ephemeris file to download.
        leaps_f : str
            JPL leap seconds file to download.
        pck_f : str
            JPL Earth orientation file to download.

        Returns
        -------
        list[pathlib.Path]
            Local paths of the three kernels, in the order given.
        """
        paths = [kernels.ensure_kernel(f) for f in (ephem_f, leaps_f, pck_f)]
        config.state["kernel_downloaded"] = True
        return paths
