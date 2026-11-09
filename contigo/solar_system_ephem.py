"""Solar System Environment container.

added: 18/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

import logging
from operator import itemgetter

import numpy as np
import numpy.typing as npt

from contigo.contigo_utils.constants import GMc
from contigo.ephemeris.base import EphemerisProvider

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

    Binning
    -------
    A GPS time ``t`` falls in bin ``round(t / tolerance)`` (``round(t)`` for
    ``tolerance=0.0``; the exact value for ``tolerance=None``). Each bin holds a
    single position per body: every time in a bin is served that one value.
    When a bin is not yet cached, the *first time in request order* that falls
    in it represents the bin: only that time is sent to the provider, once,
    and its positions are cached for the bin. Later times in the same bin, in
    the same request or in later ones, never change the cached value.

    Raises
    ------
    ValueError
        If the provider's ephemeris has no GM table, or a body has no GM entry
        in that table (the message lists the known names).
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

        # GM lookup first, so an unknown name fails before any provider call
        self.GM = self._lookup_gm(provider.ephemeris, self.bodies)

        if self._check_times(ephem_time, gps_time, utc_time):
            ephem_time = np.asarray(ephem_time, dtype=float)
            gps_time = np.asarray(gps_time, dtype=float)
            utc_time = np.asarray(utc_time)
            self._load_times(ephem_time, gps_time, utc_time)

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
        sp_gps = np.asarray(gps_time, dtype=float)
        sp_utc = np.asarray(utc_time)

        # Ensure all times are loaded
        self._load_times(sp_et, sp_gps, sp_utc)

        key = self._quantize(sp_gps)
        values = itemgetter(*key)(self._cache)
        if len(key) == 1:
            values = (values,)  # wrap bare value in tuple
        r_out = np.swapaxes(np.array(values), 0, 1)

        return sp_et, sp_gps, sp_utc, r_out

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _quantize(
        self, t: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int_ | np.float64]:
        """Quantize time using tolerance and return integer bin."""
        if self.tolerance == 0.0:
            # exact integer seconds binning
            return np.round(t).astype(np.int64)
        elif self.tolerance is None:
            return t
        return np.round(t / self.tolerance).astype(np.int64)

    def _load_times(
        self, ephem_time: np.ndarray, gps_time: np.ndarray, utc_time: np.ndarray
    ) -> None:
        """
        Load only times not already cached.
        """

        # Quantize requested times
        q_times = self._quantize(gps_time)

        # Identify the missing bins. Each is requested once, represented by the
        # first time (in request order) that falls in it; dicts keep insertion
        # order, so later times in the same bin are ignored.
        missing: dict = {}
        for q, et, utc, gps in zip(
            q_times, ephem_time, utc_time, gps_time, strict=True
        ):
            if q not in self._cache and q not in missing:
                missing[q] = (et, utc, gps)

        if not missing:
            return

        missing_qt = list(missing)
        missing_et, missing_utc, missing_gps = list(zip(*missing.values(), strict=True))

        # Call provider once for all missing bins
        r_new = self._provider(
            self.bodies,
            ephem_time=missing_et,
            gps_time=missing_gps,
            utc_time=missing_utc,
        )

        # Store per-bin slice in dictionary
        for i, t in enumerate(missing_qt):
            # r_new shape = (Nb, N_missing_bins, 3)
            self._cache[t] = r_new[:, i, :]

    @staticmethod
    def _lookup_gm(ephemeris: str, bodies: list[str]) -> np.ndarray:
        """GM (m^3/s^2) of ``bodies`` from the table of ``ephemeris``.

        The table is selected by the first five characters of the ephemeris
        name (``'de440s'`` uses ``'de440'``).

        Raises
        ------
        ValueError
            Unknown ephemeris, or a body with no GM entry in its table.
        """
        eph = str(ephemeris)[0:5]
        if eph not in GMc:
            raise ValueError(
                f"No GM table for ephemeris {ephemeris!r}; known ephemerides "
                f"(matched on the first five characters): {sorted(GMc)}"
            )
        # skip metadata entries such as 'unit'
        table = {k: v for k, v in GMc[eph].items() if not isinstance(v, str)}
        unknown = [bd for bd in bodies if bd not in table]
        if unknown:
            raise ValueError(
                f"Unknown body name(s) {unknown} for ephemeris {ephemeris!r}; "
                f"known bodies: {sorted(table)}"
            )
        return np.array([table[bd] for bd in bodies], dtype=float)

    def _check_times(self, ephem_time, gps_time, utc_time):
        return not (ephem_time is None and gps_time is None and utc_time is None)
