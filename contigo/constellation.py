"""Constellation container for managing multiple spacecraft.

added: 26/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
from typing import Sequence
import numpy as np

from .spacecraft import Spacecraft

# ==================================================================
# Constellation Container
# ==================================================================
class Constellation:
    """
    Constellation container that accepts the same inputs as Spacecraft,
    performs loading once, and then splits into individual Spacecraft
    objects using Spacecraft.split_by_id().

    This preserves all loading logic while enforcing that each member
    spacecraft is a strict single-ID Spacecraft object.
    """

    def __init__(self,
                 spacecraft: Spacecraft | Sequence[Spacecraft] | None = None,
                 **spacecraft_kwargs) -> None:
        
        # -----------------------------------------------------------
        # Case 1: List of Spacecraft → convert via iterative addition
        # -----------------------------------------------------------
        if isinstance(spacecraft, (list, tuple)):

            if not spacecraft:
                raise ValueError("Spacecraft list cannot be empty.")

            if not all(isinstance(sc, Spacecraft) for sc in spacecraft):
                raise TypeError("All elements must be Spacecraft objects.")

            # Start from first spacecraft
            base = Constellation(spacecraft=spacecraft[0])

            # Iteratively add remaining
            for sc in spacecraft[1:]:
                base += Constellation(spacecraft=sc)

            # Copy merged result into self
            self.spacecraft = base.spacecraft
            self.sspice_et = base.sspice_et
            self.sspice_gps = base.sspice_gps
            self.sc_utc = base.sc_utc

            return

        # ----------------------------------------------------------
        # Case 2: Single Spacecraft object
        # ----------------------------------------------------------
        if spacecraft is not None:
            if not isinstance(spacecraft, Spacecraft):
                raise TypeError("spacecraft must be a Spacecraft object")

            multi_sc = spacecraft
        # ----------------------------------------------------------
        # Case 3: Construct from loader kwargs
        # ----------------------------------------------------------
        elif spacecraft is None:
            # Load once using the Spacecraft loader
            multi_sc = Spacecraft(**spacecraft_kwargs)
        else:
            raise TypeError(
                "spacecraft must be Spacecraft, list[Spacecraft], or None"
            )


        # Split into individual spacecraft
        self.spacecraft: dict = multi_sc.split_by_id()

        # Union of unique times across all spacecraft in the constellation.
        # Used by EDRDensity to pre-load ephemeris for all epochs in one call.
        # sspice_gps : (M,) GPS seconds — primary sort/dedup key
        # sspice_et  : (M,) SPICE ephemeris time (TDB seconds past J2000)
        # sc_utc     : (M,) Python datetime objects in UTC
        self.sspice_gps, u_id = np.unique(multi_sc.sspice_gps, return_index=True)
        self.sspice_et = multi_sc.sspice_et[u_id]
        self.sc_utc = multi_sc.sc_utc[u_id]

    @classmethod
    def _from_add(cls, spacecraft_dict, sspice_et, sspice_gps, sc_utc):
        obj = cls.__new__(cls)
        obj.spacecraft = spacecraft_dict
        obj.sspice_et = sspice_et
        obj.sspice_gps = sspice_gps
        obj.sc_utc = sc_utc
        return obj

    # --------------------------------------------------------------
    @property
    def ids(self) -> list:
        return list(self.spacecraft.keys())

    def __getitem__(self, key):
        return self.spacecraft[key]

    def __iter__(self):
        return iter(self.spacecraft.values())

    def __len__(self):
        return len(self.spacecraft)

    def __repr__(self) -> str:
        return f"Constellation(n_spacecraft={len(self)}, ids={self.ids})"
    
    def __add__(self, other: "Constellation") -> "Constellation":
        if not isinstance(other, Constellation):
            return NotImplemented
        
        # Prevent duplicate spacecraft IDs
        overlap = set(self.ids).intersection(other.ids)
        if overlap:
            raise ValueError(
                f"Cannot merge constellations. Duplicate spacecraft IDs found: {overlap}"
            )
        
        merged_dict = self.spacecraft | other.spacecraft
        merged_et = self._merge_time_array(self.sspice_et, other.sspice_et)
        merged_gps = self._merge_time_array(self.sspice_gps, other.sspice_gps)
        merged_utc = self._merge_time_array(self.sc_utc, other.sc_utc)

        return Constellation._from_add(
            spacecraft_dict=merged_dict,
            sspice_et=merged_et,
            sspice_gps=merged_gps,
            sc_utc=merged_utc)
    
    def __iadd__(self, other: "Constellation") -> "Constellation":
        if not isinstance(other, Constellation):
            return NotImplemented

        overlap = set(self.ids).intersection(other.ids)
        if overlap:
            raise ValueError(
                f"Cannot merge constellations. Duplicate spacecraft IDs found: {overlap}"
            )

        self.spacecraft.update(other.spacecraft)

        self.sspice_et = self._merge_time_array(self.sspice_et, other.sspice_et)
        self.sspice_gps = self._merge_time_array(self.sspice_gps, other.sspice_gps)
        self.sc_utc = self._merge_time_array(self.sc_utc, other.sc_utc)

        return self

    @staticmethod
    def _merge_time_array(a, b):
        """
        Merge two time arrays safely:
        - Handles numpy arrays or lists
        - Removes duplicates
        - Returns sorted unique array
        """
        if a is None:
            return b
        if b is None:
            return a

        a_arr = np.asarray(a)
        b_arr = np.asarray(b)

        merged = np.concatenate((a_arr, b_arr))
        return np.unique(merged)
