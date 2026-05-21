"""Ephemeris provider using Orekit.

added: 04/03/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
import numpy as np
import numpy.typing as npt

import contigo.config as config
if config.state['orekit_loaded'] is False:
    from contigo.contigo_utils import orekit_utils
    orekit_utils.start_orekit()

from java.util import ArrayList

from contigo.ephemeris.base import EphemerisProvider

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.contigo.orekit_utils import EphemerisBatchHelper

class OrekitEphem(EphemerisProvider):
    """
    Orekit-backed ephemeris provider.
    """

    def __init__(self, 
                 ephemeris: str='de440s'):
        
        self.ephemeris= ephemeris

    def __call__(self, body: list[str],
                 utc_time: np.ndarray | None = None,
                 gps_time: None = None,
                 ephem_time: None = None,
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
        This provider only uses utc_time for its internal logic, but the interface 
        allows for flexibility in case future providers need to use different
        time systems.
        """
        utc = TimeScalesFactory.getUTC()
        first_dt = utc_time[0]
        ref_date = AbsoluteDate(first_dt.year, first_dt.month, first_dt.day,
                                        first_dt.hour, first_dt.minute,
                                        float(first_dt.second), utc) 
        offsets = np.array(
            [(t - first_dt).total_seconds() for t in utc_time],
            dtype=np.float64)

        r_body = EphemerisBatchHelper.getBodyECEF(ref_date, 
                                                  offsets, 
                                                  ArrayList(body))
        
        return np.array(r_body, dtype=np.float64)