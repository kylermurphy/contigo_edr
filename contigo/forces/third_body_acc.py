"""Derive third body accelerations for an Earth orbiting spacecraft.

added: 17/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
import posixpath
import urllib.parse
import logging

from os import path
from datetime import datetime, timezone
from dateutil import tz

import pandas as pd
import numpy as np
import numpy.typing as npt
import spiceypy as spice

import contigo.contigo_utils.utils as utils
import contigo.config as config

from contigo.ephemeris.spice_ephem import SPICEEphem
from contigo.forces.base import ForceModel
from contigo.solar_system_ephem import SolarSystemEnvironment

from .tba_utils import tba_pairwise_numba
from contigo.contigo_utils.constants import GMc
from contigo.constellation import Constellation

logger = logging.getLogger(__name__)

class ThirdBodyAcc:
    """Deriving Third Body Acceleration using JPL SPICE
    """
    def __init__(self, 
                 spos: npt.ArrayLike | None = None,
                 stime: npt.ArrayLike | None = None,
                 body: npt.ArrayLike | None = None,
                 GM: npt.ArrayLike | None = None,
                 scale: str | None = None,
                 ephemeris: str='de440s'):
        """Initialize the ThirdBodyAcc Class for deriving accelleration from third
        bodies such as the Sun and Moon.

        Position is a (n,3) array that needs to be in Earth Centered Earth Fixed
        (ECEF) coordinate frame. Units are meters.

        The body GM constants are in units of m^3/s^2.

        Uses the SPICE (Spacecraft, Planet, Instrument, C-matrix, Events) observation 
        geometry information system and Kernels from JPLs Navigation and Ancillary 
        Information Facility (NAIF) (https://naif.jpl.nasa.gov/naif/index.html).
        
        Required JPL SPICE kernels are downloaded as needed and when changed on the web.

        Parameters
        ----------
        spos : npt.ArrayLike (n,3), optional
            An array of spacecraft positions (ECEF - meters),
            by default np.array([6_771_000.0,0,0],ndmin=2)
        stime : npt.ArrayLike (n), optional
            An array of spacecraft times for which the position of body are retrieved.
            The scale of the time is needed in order to derive JPL SPICE ephemeris 
            time (ET). By default pd.Series(pd.to_datetime('2020-01-01')).
        body : npt.ArrayLike, optional
            A list of bodies for which to calculate accelerations at the location 
            of a spacecraft (spos), by default ['SUN'].
        GM : npt.ArrayLike | None, optional
            Mass parameters for the bodies in body. Should be in the same order as body.
            If nothing is passed they are loaded from the constants module which uses
            JPLs de440 mass parameters, by default None.
        scale : str | None, optional
            Time scale of stime; allowed values are 'GPS','TAI','UTC','ET','TDB'.
            By default None but one is required and a Value error will be thrown if it 
            is not passed or if it is not one of the allowed values.
        ephemeris : str, optional
            JPL SPICE ephemeris file to use; allowed values are de440 and de440s. 's'
            denotes the smaller version which covers a shorter time frame. By default 
            'de440s'

        Raises
        ------
        ValueError
            ephemeris not in allowed ephemeris
        ValueError
            scale not in allowed scales
        """
        # set defaults
        if spos is None:
            spos = np.array([[6_771_000.0, 0.0, 0.0]])
        if stime is None:
            stime = pd.Series(pd.to_datetime("2020-01-01"))
        if body is None:
            body = ["SUN"]

        # allowed ephemeris to load
        allowed_eph = ['de440','de440s']
        if ephemeris in allowed_eph:
            eph_sh = ephemeris[0:5]
        else:
            raise ValueError(f'ephemeris must be on of allowed {allowed_eph}')

        # allowed time scales for getting third body accelerations
        allowed_scales = ['GPS','TAI','UTC','ET','TDB']
        if not scale:
            raise ValueError(f'scale must be one of {allowed_scales}')
        
        scale = scale.upper()
        if scale not in allowed_scales:
            raise ValueError(f'Incorrect scale {scale}, scale must be one of {allowed_scales}')

        self.spos = np.asarray(spos, dtype=float)
        if self.spos.ndim != 2 or self.spos.shape[1] != 3:
            raise ValueError("spos must have shape (N,3)")

        self.stime = pd.to_datetime(stime)
        if self.stime.shape[0] != self.spos.shape[0]:
            raise ValueError("spos and stime must have be (N,3) and (N)")

        self.body = [bd.upper() for bd in body]
        
        self.scale = scale
        if GM is None:
            self.GM = np.array([GMc[eph_sh][bd] for bd in self.body],dtype=float)
        else:
            self.GM = np.asarray(GM, dtype=float)

        if len(self.body) != len(self.GM):
            raise ValueError("body and GM must be same length")
        self.ephemeris = ephemeris
        #attributes used later
        self.bd_ecef = None
        self.bd_acc = None

    def calc_tba(self):
        """Derives third body accelerations from spacecraft positions for solar
        system bodies.

        Uses the SPICE (Spacecraft, Planet, Instrument, C-matrix, Events) observation 
        geometry information system and Kernels from JPLs Navigation and Ancillary 
        Information Facility (NAIF).
        """
        # calculate the seconds past j2000 to convert
        # to SPICE ET Ephemeris time (in the SPICE system,
        # this is equivalent to TDB time)
        if self.stime.tzinfo is not None and self.stime.tzinfo != timezone.utc:
            print(str(self.stime.tzinfo))
            raise ValueError('stime should be time zone naive or UTC')
        

        ephem = SPICEEphem(ephemeris=self.ephemeris)

        # set all needed attributes
        if self.scale == 'UTC':
            t_str = pd.to_datetime(np.array(self.stime)).strftime('%d %b %Y %H:%M:%S.%f')
            et = np.array([spice.utc2et(sp_in) for sp_in in t_str]) 
        else:
            j2000 = pd.Timestamp('2000-01-01 12:00:00')
            spj2000 = ((self.stime - j2000).total_seconds()).to_list()
            et = np.array([spice.unitim(sp_in,self.scale,'ET') for sp_in in spj2000])

        et = np.array(et)

        _, bd_ecef = ephem(body=self.body,et=et)

        bd_acc = tba_pairwise_numba(self.spos, bd_ecef, self.GM)

        self.bd_acc = bd_acc
        self.bd_ecef = bd_ecef

    def get_tba(self):
        """Get the Third Body Accelerations.

        Returns
        -------
        np.Array
            Third body accelerations
        """
        if self.bd_acc is None:
            raise RuntimeError("calc_tba() must be called first")
        return self.bd_acc

    def get_body_pos(self):
        """Get the Third Body ECEF Positions.

        Returns
        -------
        np.Array
            Third body positions in ECEF coordinates.
        """
        if self.bd_ecef is None:
            raise RuntimeError("calc_tba() must be called first")
        return self.bd_ecef


class ThirdBody():
    """
    Third-body gravity force operating on invdividual satellites in a Constellation
    object.
    """

    name: str = "ThirdBodyAcceleration"

    def __init__(self,
                 body=None,
                 GM=None,
                 ephemeris: str = "de440s",):
        """Third body gravity acting on individual satellites in a Constellation
        
        Parameters
        ----------
        body : npt.ArrayLike, optional
            A list of bodies for which to calculate accelerations at the location 
            of a spacecraft (spos), by default ['SUN'].
        GM : npt.ArrayLike | None, optional
            Mass parameters for the bodies in body. Should be in the same order as body.
            If nothing is passed they are loaded from the constants module which uses
            JPLs de440 mass parameters, by default None.
        ephemeris : str, optional
            JPL SPICE ephemeris file to use; allowed values are de440 and de440s. 's'
            denotes the smaller version which covers a shorter time frame. By default 
            'de440s'

        """        
        self.body = body
        self.GM = GM
        self.ephemeris = ephemeris

    def acceleration(self, 
                     constellation: Constellation
                     ) -> dict[str, npt.NDArray[np.float64]]:
        """Derive third body accellerations.

        Use ThirdBodyAcc to derive accelerations for satellites in a Constellation 
        object.

        Constellation holds the state and time scale of all satellites.

        Initialization defines the ephemeris to use and the solar system bodies

        Parameters
        ----------
        constellation : Constellation
            Constellation container of Spacecraft objects.

        Returns:
            dict[spacecraft_id] -> (N,3)
        """

        acc_dict = {}

        for sc_id, sc in constellation.spacecraft.items():

            tba = ThirdBodyAcc(
                spos=sc.state_ecef[:, 0:3],
                stime=sc.stime,
                body=self.body,
                GM=self.GM,
                scale=sc.tscale,
                ephemeris=self.ephemeris,
            )

            tba.calc_tba()
            acc_dict[sc_id] = tba.get_tba()

        return acc_dict

    def potential(self, 
                  constellation: Constellation
                  ) -> dict[str, npt.NDArray[np.float64]]:
        raise NotImplementedError("Not implemented for ThirdBodyAcc.")
    
class ThirdBodyEnv(ForceModel):
    """
    Third-body gravity force operating on invdividual satellites in a Constellation
    object.
    """

    name: str = "ThirdBodyAcceleration"

    def __init__(self):
        pass

    def acceleration(self, 
                     constellation: Constellation, 
                     solarsys_env: SolarSystemEnvironment
                     ) -> dict[str, npt.NDArray[np.float64]]: 
        """Derive third body acceleration for a Constellation of Spacecraft from the
        SolarSystemEnvironment

        Parameters
        ----------
        constellation : Constellation
            Constellation container with the spcaecraft state, physical properties,
            spacecraft IDs, and time arrays.
        solarsys_env : SolarSystemEnvironment
            Solar System Environment container to load the ephemeris of solar system 
            bodies. The ephemeris is loaded for the derivation of the third body 
            accelerations.

        Returns
        -------
        dict[str, npt.NDArray[np.float64]]
            A dictionary keyed by spacecraft ID. The array is an (B,N,3) array where 
            B is the number of bodies in the SolarSystemEnvironment, N is the number of
            spacecraft states, and 3 is the x,y,z components of the third body
            acceleration. The order of the bodies is the same as the order of the bodies
            in the SolarSystemEnvironment.
        """
        acc_dict = {}

        for sc_id, sc in constellation.spacecraft.items():

            _,_, _, r_pos = solarsys_env.get_ephem(ephem_time=sc.sspice_et,
                                                 gps_time=sc.sspice_gps,
                                                 utc_time=sc.sc_utc)

            tba = tba_pairwise_numba(sc.state_ecef[:, 0:3], r_pos, solarsys_env.GM)
            acc_dict[sc_id] = tba

        return acc_dict

    def potential(self,
                  constellation: Constellation
                  ) -> dict[str, npt.NDArray[np.float64]]:
        """Not implemented for ThirdBodyAcc, only acceleration is calculated."""
        raise NotImplementedError("Not implemented for ThirdBodyAcc.")