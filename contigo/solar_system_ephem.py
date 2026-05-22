"""Solar System Environment container.

added: 18/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
import logging
from operator import itemgetter
import numpy as np
import numpy.typing as npt

from contigo.ephemeris.base import EphemerisProvider
from contigo.contigo_utils.constants import GMc

logger = logging.getLogger(__name__)

class SolarSystemEnvironment:
    """
    High-performance solar system ephemeris cache.

    Design
    ------
    Bodies are static after initialization
    Tolerance is static after initialization
    Internally uses a dictionary keyed by quantized time
    O(1) lookup instead of O(N²) tolerance scans
    Cached arrays rebuilt lazily only if needed
    """

    def __init__(
        self,
        bodies: npt.NDArray[np.str_],
        tolerance: float | None,
        provider: EphemerisProvider,
        ephem_time: npt.NDArray[np.float64] | None = None,
        gps_time: npt.NDArray[np.float64] | None = None,
        utc_time: npt.NDArray[np.datetime64] | None = None,
    ) -> None:

        self.bodies = [b.upper() for b in bodies]
        if tolerance is None:
            self.tolerance = None
        else:
            self.tolerance = float(tolerance)
        self._provider = provider

        # Internal dictionary cache
        # key   -> quantized time (float)
        # value -> (Nb, 3) position array
        self._cache: dict[float, np.ndarray] = {}

        if self._check_times(ephem_time, gps_time, utc_time):
            ephem_time = np.asarray(ephem_time, dtype=float)
            gps_time = np.asarray(gps_time,dtype=float)
            utc_time = np.asarray(utc_time)
            self._load_times(ephem_time,gps_time,utc_time)

        eph = provider.ephemeris[0:5]

        self.GM = np.array([GMc[eph][bd] for bd in self.bodies],dtype=float)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def get_ephem(
        self,
        ephem_time: npt.NDArray[np.float64],
        gps_time: npt.NDArray[np.float64],
        utc_time: npt.NDArray[np.datetime64],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve ephemeris data for requested times using cached or newly
        computed values.

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
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (ephem_time, gps_time, utc_time, positions)

        positions : np.ndarray
            Array of shape (n_bodies, N, 3) containing position vectors
            of celestial bodies in the configured reference frame.

        Notes
        -----
        Uses time quantization to reduce redundant ephemeris lookups.
        """
        sp_et = np.asarray(ephem_time, dtype=float)
        sp_gps = np.asarray(gps_time,dtype=float)
        sp_utc = np.asarray(utc_time)

        # Ensure all times are loaded
        self._load_times(sp_et, sp_gps, sp_utc)

        key = self._quantize(sp_gps)
        r_out = np.array(itemgetter(*key)(self._cache))
        r_out = np.swapaxes(r_out,0,1)

        return sp_et, sp_gps, sp_utc, r_out

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _quantize(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.int_ | np.float64]:
        """Quantize time using tolerance and return integer bin."""
        if self.tolerance == 0.0:
            # exact integer seconds binning
            return np.round(t).astype(np.int64)
        elif self.tolerance is None:
            return t
        return np.round(t / self.tolerance).astype(np.int64)

    def _load_times(self, 
                    ephem_time: np.ndarray,
                    gps_time: np.ndarray,
                    utc_time: np.ndarray) -> None:
        """
        Load only times not already cached.
        """

        # Quantize requested times
        q_times = self._quantize(gps_time)

        # Identify which quantized times are missing
        missing = [
                   [i,et,utc,gps]
                   for i, et, utc, gps in zip(q_times,ephem_time,utc_time,gps_time) 
                   if i not in self._cache
                   ]

        if not missing:
            return

        missing_qt, missing_et, missing_utc, missing_gps = list(zip(*missing))

        # Call provider once for all missing times
        r_new = self._provider(self.bodies,
                               ephem_time=missing_et,
                               gps_time=missing_gps,
                               utc_time=missing_utc)


        # Store per-time slice in dictionary
        for i, t in enumerate(missing_qt):
            # r_new shape = (Nb, Nt_missing, 3)
            self._cache[t] = r_new[:, i, :]

    def _check_times(self, ephem_time, gps_time, utc_time):
        if isinstance(ephem_time, type(None)) and isinstance(gps_time, type(None)) and isinstance(utc_time, type(None)):
            return False
        else:
            return True