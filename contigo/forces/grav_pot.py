"""Derive gravitational potential of Earth for a set of ECEF coordinates.

added: 03/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

import logging
import os.path

import numpy as np
import numpy.typing as npt

import contigo.config as config
from contigo.constellation import Constellation
from contigo.forces.base import ForceModel
from contigo.solar_system_ephem import SolarSystemEnvironment

from .grav_utils import get_potential_batch, read_icgem_coeff

logger = logging.getLogger(__name__)


class GravPot:
    """Class to derive gravitational potential for a set of ECEF coordinates."""

    def __init__(
        self,
        r: npt.ArrayLike = np.array(6_771_000.0, ndmin=1),
        lat: npt.ArrayLike = np.array(0, ndmin=1),
        lon: npt.ArrayLike = np.array(np.pi, ndmin=1),
        pot_file: str = "EIGEN-2.gfc",
        lmax: int = 50,
    ):
        """Initialize the GravPot Class to calculate Earth gravitational potential.

        Position (r, lat, lon) needs to be in an Earth Centered Earth Fixed
        (ECEF) coordinate frame.
        Units are meters and radians for position (r, lat, lon).
        The Gravitational Potential is calculated in units of m^2/s^2.

        Parameters
        ----------
        r : npt.ArrayLike, optional
            An array of radial positions (ECEF - meters),
            by default np.array(6_771_000.0,ndmin=1)
        lat : npt.ArrayLike, optional
            An array of latitude positions (ECEF - radians),
            by default np.array(0,ndmin=1)
        lon : npt.ArrayLike, optional
            An array of longitude positions (ECEF - radians),
            by default np.array(np.pi,ndmin=1)
        pot_file : str, optional
            ICGEM potential file used to derive the gravitational
            potential, uses .grav_utils.read_icgem_coeff which returns
            clm, slm, and metadata, by default 'EIGEN-2.gfc'
        lmax : int, optional
            Maximum degree/order for deriving the gravitational potential.
            If lmax is larger than the max degree/order of the loaded potential
            file then lmax is set to the file's maximum (on first load and when
            the coefficients come from the ``config.state`` cache); None or 0
            selects the file's maximum, by default 50
        """
        r = np.asarray(r)
        if r.ndim == 0:
            r = r.reshape(1)
        lat = np.asarray(lat)
        if lat.ndim == 0:
            lat = lat.reshape(1)
        lon = np.asarray(lon)
        if lon.ndim == 0:
            lon = lon.reshape(1)

        # Initial Variables For Potential
        self.r = r
        self.lat = lat
        self.lon = lon

        self.pot_file = pot_file
        self.lmax = lmax
        self.gravpot = None

        # Resolve the potential file: as given, else in the package data dir
        if os.path.isfile(pot_file):
            pass
        elif os.path.isfile(os.path.join(config.DATA_DIR, self.pot_file)):
            self.pot_file = os.path.join(config.DATA_DIR, self.pot_file)
        else:
            raise ValueError(f"Potential file {self.pot_file} cannot be found.")

        # The coefficient cache is keyed on the resolved full path, so two
        # different files with the same basename never share coefficients.
        pot_key = os.path.realpath(self.pot_file)

        # Modeled Field Variables Which are Loaded
        if (
            config.state["pot_coef_loaded"] is True
            and config.state["pot_file"] == pot_key
        ):
            logger.info(
                "Loading potential coeffecients from current state which used %s.",
                config.state["pot_file"],
            )

            self.clm = config.state["pot_clm"]
            self.slm = config.state["pot_slm"]
            self.r0 = config.state["pot_r0"]
            self.GM = config.state["pot_GM"]
            file_lmax = config.state.get("pot_lmax")
            if file_lmax is None:
                file_lmax = self.clm.shape[0] - 1
            self.lmax = self._clamp_lmax(self.lmax, file_lmax)
        else:
            self.load_coeff()

    @staticmethod
    def _clamp_lmax(lmax, file_lmax):
        """Return lmax limited to the maximum degree of the potential file.

        Parameters
        ----------
        lmax : int or None
            Requested maximum degree/order. None or 0 selects the file's lmax.
        file_lmax : int
            Maximum degree/order available in the potential file.

        Returns
        -------
        int
            The lmax to use.
        """
        if not lmax:
            return file_lmax
        if lmax > file_lmax:
            logger.warning(
                "Defined lmax is to large. Setting lmax to that in the "
                "potential file (%d).",
                file_lmax,
            )
            return file_lmax
        return lmax

    def load_coeff(self):
        """Load ICGEM clm, slm potential coefficients from ``self.pot_file``.

        Clamps ``self.lmax`` to the file's maximum degree and stores the
        coefficients in ``config.state``, keyed on the resolved full path of
        the file.
        """
        logger.info("Loading potential file %s", self.pot_file)

        self.clm, self.slm, cs_meta = read_icgem_coeff(self.pot_file)
        self.r0 = cs_meta["r0"]
        self.GM = cs_meta["GM"]
        # check the lmax of the potential file
        self.lmax = self._clamp_lmax(self.lmax, cs_meta["lmax"])

        # Set potential state variables
        # so we don't have to keep loading
        # coeffecients
        config.state["pot_file"] = os.path.realpath(self.pot_file)
        config.state["pot_coef_loaded"] = True
        config.state["pot_clm"] = self.clm
        config.state["pot_slm"] = self.slm
        config.state["pot_r0"] = self.r0
        config.state["pot_GM"] = self.GM
        config.state["pot_lmax"] = cs_meta["lmax"]

    def calc_pot(self):
        """Derive Potential.

        Potential is derived using normalize Legendre Coefficients from
        pyshtools.

        .grav_utils.get_potential is used to initially derive all required
        values which are then passed to .grav_utils._get_potential_numba_core
        which uses numba and jit to improve the performace of the calculation.

        The Gravitational Potential is calculated in units of m^2/s^2.

        Raises
        ------
        ValueError
            If potential coefficients haven't been loaded.
        ValueError
            If the size of the position values, r, lat, lon are not the same.
        """
        # check if coefficients have been loaded
        if self.clm is None:
            raise ValueError("No coeffecients have been loaded, use load_coef( )")
        # check to make r, lat, lon are the same size
        if len(self.r) != len(self.lat) or len(self.r) != len(self.lon):
            raise ValueError("r, lat, and lon must be same length")

        self.gravpot = get_potential_batch(
            self.r,
            self.lat,
            self.lon,
            self.clm,
            self.slm,
            self.GM,
            self.r0,
            lmax=self.lmax,
        )

    def get_pot(self):
        """Return Potential.

        Returns
        -------
        np.array
            Array of potetial in m^2/s^2

        Raises
        ------
        ValueError
            If ``calc_pot()`` has not been called.
        """
        if self.gravpot is None:
            raise ValueError("Potential needs to be calculated; call calc_pot() first")
        return self.gravpot


class EarthPotential(ForceModel):
    """
    Earth Gravatational Potential ForceModel implementation using the GravPot class.
    """

    name: str = "EarthGravitationalPotential"

    def __init__(
        self,
        pot_file: str = "EIGEN-2.gfc",
        lmax: int = 50,
    ):
        """
        Initialize the EarthPotential ForceModel.

        Parameters
        ----------
        pot_file : str, optional
            ICGEM potential file to use, by default 'EIGEN-2.gfc'
        lmax : int, optional
            Maximum degree of the spherical harmonic expansion, by default 50
        """
        self.pot_file = pot_file
        self.lmax = lmax

    def acceleration(
        self, constellation: Constellation, solarsys_env: SolarSystemEnvironment
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Not implemented for EarthPotential, only potential is calculated"""
        raise NotImplementedError("Not implemented for EarthPotential.")

    def potential(
        self, constellation: Constellation
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Return the potential for each Spacecraft in a Constellation object

        Parameters
        ----------
        constellation : Constellation
            A Constellation object containing one or more Spacecraft objects for
            which the potential will be calculated.

        Returns
        -------
        dict[str, npt.NDArray[np.float64]]
           A dictionary keyed by spacecraft ID. Each value of the key is an array
           corresponding to the potential for each position of the Spacecraft state.
        """
        pot_dict = {}

        for sc_id, sc in constellation.spacecraft.items():
            sc_sph = sc.spherical()

            ep = GravPot(
                r=sc_sph[:, 0],
                lat=sc_sph[:, 1],
                lon=sc_sph[:, 2],
                pot_file=self.pot_file,
                lmax=self.lmax,
            )

            ep.calc_pot()
            pot_dict[sc_id] = ep.get_pot()

        return pot_dict
