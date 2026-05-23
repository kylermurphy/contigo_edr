"""Ephemeris base protocol and common logic class.

Simplifies and standardizes the inteface for ephemeris providers.

added: 09/04/2026  Kyle Murphy <kylemurphy.spacephys@gmail.com>

"""

from typing import Protocol, runtime_checkable

import numpy as np

@runtime_checkable
class EphemerisProvider(Protocol):
    """
    Minimal ephemeris provider interface.

    Any ephemeris backend (SPICE, Orekit, etc.) must implement
    a callable interface returning body positions.

    The SolarSystemEnvironment will treat the provider as a black box
    and only rely on this contract.

    Notes
    -----
    - This intentionally enforces *minimal coupling*
    - Allows swapping SPICE ↔ Orekit without changing environment logic
    - Keeps batch/vectorization decisions inside provider
    """

    def __call__(self,
                 body: list[str],
                 utc_time: np.ndarray | None = None,
                 gps_time: None = None,
                 ephem_time: None = None,
                ) -> np.ndarray:
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

        Notes:
        ------
        - The provider can choose to use any combination of time inputs
          (ephem_time, gps_time, utc_time) as needed for its internal 
          logic.
        """
        ...