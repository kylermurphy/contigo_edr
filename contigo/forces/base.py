"""Force model protocol that all forces must follow.

Simplifies and standardizes the inteface for all forces. This allows easy accees and
easy addition of new forces to the EDR and EFD calculations.

Wrappers can be written to convert existing force models to this protocol,
and new force models can be written to follow this protocol from the start.

added: 18/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from contigo.constellation import Constellation
from contigo.solar_system_ephem import SolarSystemEnvironment


@runtime_checkable
class ForceModel(Protocol):
    """
    All force models operate on a Constellation.

    A single Spacecraft should be represented as a
    Constellation with one member.
    """

    name: str

    def acceleration(
        self, constellation: Constellation, solarsys_env: SolarSystemEnvironment
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Compute acceleration for all spacecraft in a constellation.

        Parameters
        ----------
        constellation : Constellation
            Collection of spacecraft objects with state vectors.
        solarsys_env : SolarSystemEnvironment
            Ephemeris provider and cache.

        Returns
        -------
        dict
            Dictionary mapping spacecraft IDs to acceleration arrays
            of shape (N, 3) or (n_bodies, N, 3).
        """
        ...

    def potential(
        self, constellation: Constellation
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Compute potential for all spacecraft in a constellation.

        Parameters
        ----------
        constellation : Constellation
            Collection of spacecraft objects with state vectors.

        Returns
        -------
        dict
            Dictionary mapping spacecraft IDs to potential arrays
            of shape (N,).
        """
        ...
