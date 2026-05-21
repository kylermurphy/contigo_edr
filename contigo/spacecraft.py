"""Spacecraft conatiner class for managin individual spacecraft states, 
physical properties, and IDs.

added: 18/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import os
import glob

import numpy as np
import numpy.typing as npt
import pandas as pd

import spiceypy as spice

from contigo.contigo_utils import time_utils
from contigo.contigo_utils import utils

@dataclass
class Spacecraft:
    """Spacecraft class for loading and storing spacecraft state info.

    This is an extended container class for the contigo module

    Raw inputs may be provided directly (state, time) OR loaded
    from one or more files on disk. Internal state is always strict
    and guaranteed after loading. A simplified container class 
    SpacecraftState is used to store the finalized internal state. 

    The state is assumed to be position [x,y,z] (m), velocity [vx,vy,vz] (m/s) in ECEF
    and spacecraft physical properties and ID. Internal state is always stored in SI units.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        FileNotFoundError: _description_
        NotImplementedError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """    

    # ------------------------------------------------------------------
    # Raw user inputs (flexible)
    # ------------------------------------------------------------------
    state: npt.ArrayLike | None = None        # (N,6) [x,y,z,vx,vy,vz]
    time: npt.ArrayLike | None = None         # (N,)
    sc_id_input: npt.ArrayLike | None = None  # (N,)
    tscale_input: str | None = None          # ['GPS','TAI','UTC','ET','TDB']
    unit_input: str = 'km'

    # Spacecraft physical properties (scalar or (N,))
    cd: float | npt.ArrayLike | None = None
    drag_area: float | npt.ArrayLike | None = None
    sc_mass: float | npt.ArrayLike | None = None
    cr: float | npt.ArrayLike | None = None
    srp_area: float | npt.ArrayLike | None = None

    # Optional file input
    state_file: str | Path | Iterable[str | Path] | None = None

    # ------------------------------------------------------------------
    # Loader configuration (schema mapping)
    # ------------------------------------------------------------------
    time_col: str = "time"
    x_col: str = "x"
    y_col: str = "y"
    z_col: str = "z"
    vx_col: str = "vx"
    vy_col: str = "vy"
    vz_col: str = "vz"
    sc_id_col: str | None = None
    sc_fn_slc: slice | None = None

    # Optional physical-property columns
    cd_col: str | None = None
    drag_area_col: str | None = None
    sc_mass_col: str | None = None
    cr_col: str | None = None
    srp_area_col: str | None = None

    # ------------------------------------------------------------------
    # Internal normalized state (strict, always SI: m and m/s)
    # ------------------------------------------------------------------
    state_ecef: npt.NDArray[np.float64] = field(init=False)  # (N,6)
    stime: pd.DatetimeIndex = field(init=False)              # (N,)
    sspice_et: npt.NDArray[np.float64] = field(init=False)   # (N,)
    sspice_gps: npt.NDArray[np.float64] = field(init=False)  # (N,)
    sc_utc: npt.NDArray = field(init=False)  # (N,)
    tscale: str = field(init=False)                          # validated time scale
    sc_id: npt.NDArray = field(init=False)                   # (N,)
    unique_ids: npt.NDArray = field(init=False)

    # Normalized spacecraft properties (always (N,))
    cd_arr: npt.NDArray = field(init=False)
    drag_area_arr: npt.NDArray = field(init=False)
    sc_mass_arr: npt.NDArray = field(init=False)
    cr_arr: npt.NDArray = field(init=False)
    srp_area_arr: npt.NDArray = field(init=False)

    def __post_init__(self):
        """Post initialization of the Spacecraft class

        Based on input load data from arrays or files.

        Raises:
            ValueError: if arrays are provide both a spacecraft state array and 
            time array must passed. 
            ValueError: either (state, time) arrays must be passed or a container of 
            files.
        """        
        self._validate_timescale()
        self.unit = self.unit_input.lower()

        if self.state is not None or self.time is not None:
            if self.state is None or self.time is None:
                raise ValueError("Both state and time must be provided together")
            self.load_from_arrays(self.state, self.time, self.sc_id_input)
        else:
            if self.state_file is None:
                raise ValueError("Either (state, time) or state_file must be provided")
            self.load_from_file(self.state_file)
        
        # normalize time
        self._normalize_time()
        # normalize units
        self._normalize_units()

    def load_from_arrays(self, state: 
                         npt.ArrayLike,
                         time: npt.ArrayLike,
                         sc_id: npt.ArrayLike | None = None,) -> None:
        """Load spacecraft state from arrays passed in creation.

        Args:
            state (npt.ArrayLike): (N,6) spacecraft state [x,y,z,vx,vy,vz]

            time (npt.ArrayLike): (N,) time associated with spacecraft state

            sc_id (npt.ArrayLike | None, optional): spacecraft IDs to parse. Defaults to None.

        Raises:
            ValueError: make sure spacecraft state is (N,6) containing position [x,y,z] and
            velocity [vx,vy,vz].

            ValueError: spacecraft state and time must have the same number of elements in 
            the dimensions.

            ValueError: if passed spacecraft IDs must (N,)
        """
        # clear any cached data
        self._clear_cache() 

        s = np.asarray(state, dtype=float)
        if s.ndim != 2 or s.shape[1] != 6:
            raise ValueError("state must have shape (N,6)")

        t = pd.to_datetime(np.array(time), utc=False)
        if len(t) != s.shape[0]:
            raise ValueError("state and time must have the same length")

        # Spacecraft ID handling
        if sc_id is not None:
            sc = np.asarray(sc_id)
            if sc.ndim != 1 or len(sc) != len(t):
                raise ValueError("sc_id must be 1D and same length as time")
            sc_id_arr = pd.Series(sc).astype("category").to_numpy()
        else:
            sc_id_arr = pd.Series(
                np.full(len(t), "NO_ID", dtype=object)
            ).astype("category").to_numpy()

        # Commit normalized state
        self.state_ecef = s
        self.stime = t
        self.sc_id = sc_id_arr
        self.unique_ids = np.unique(self.sc_id)

        # Normalize physical properties
        self._normalize_properties(len(t), df=None)

    # ------------------------------------------------------------------
    # Loader / normalizer (file-based)
    # ------------------------------------------------------------------
    def load_from_file(self,
                       state_file: str | Path | Iterable[str | Path],
                       loader: str | None = None,
                       read_kwargs: dict | None = None,) -> None:
        """Load spacecraft state from one or more files.

        Args:
            state_file (str | Path | Iterable[str  |  Path]): Files to load into spacecraft
            state. Specific columns are required to be loaded (time, x, y, z, vx, vy, vs).
            optional columns are (spacecraft id, coeffecient of drag, drag area, spacecraft
            mass, coeffecient of reflection, SRP area).

            loader (str | None, optional): Specify the loader to use to when parseing the
            state files (state_file). If it is None then the loader is inferred from the
            file extension. The loaders which can be used are pandas.read_csv(),
            pandas.read_hdf(), or contigo.utils.df_sp3() Defaults to None.

            read_kwargs (dict | None, optional): Keyword arguments to pass to the loaders.
            Defaults to None.

        Raises:
            ValueError: Specific columns must be defined when reading data.
            (t, x, y, z, vx, vy, vz).

            ValueError: Spacecraft state must be (N,6).

            ValueError: First dimension of Spacecraft state and time must be N.
        """
        # clear any cached data
        self._clear_cache()    

        files = self._expand_files(state_file)
        frames: list[pd.DataFrame] = []

        for file in files:
            df = self._load_table(file, loader, read_kwargs)
            if self.sc_id_col is not None and self.sc_id_col.lower() == 'filename':
                base_file = os.path.basename(file)
                if isinstance(self.sc_fn_slc, slice):
                    base_file = base_file[self.sc_fn_slc]
                df['filename'] = base_file
            frames.append(df)

        df_all = pd.concat(frames, ignore_index=True)

        required = {
            self.time_col,
            self.x_col, self.y_col, self.z_col,
            self.vx_col, self.vy_col, self.vz_col,
        }

        missing = required - set(df_all.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Normalize time
        t = pd.to_datetime(df_all[self.time_col].to_numpy(), utc=False)

        # Normalize state vector
        s = df_all[[
            self.x_col, self.y_col, self.z_col,
            self.vx_col, self.vy_col, self.vz_col,
        ]].to_numpy(dtype=float)

        if s.ndim != 2 or s.shape[1] != 6:
            raise ValueError("State data must have shape (N,6)")
        if len(t) != s.shape[0]:
            raise ValueError("Time and state must have equal length")

        # Spacecraft ID handling
        if self.sc_id_col is not None and self.sc_id_col in df_all.columns:
            sc_id_arr = df_all[self.sc_id_col].astype("category").to_numpy()
        elif self.sc_id_input is not None:
            sc_id_arr = pd.Series(
                np.full(len(t), self.sc_id_input, dtype=object)).astype("category").to_numpy()
        else:
            sc_id_arr = pd.Series(
                np.full(len(t), "NO_ID", dtype=object)).astype("category").to_numpy()

        # Commit normalized state
        self.state_ecef = s
        self.stime = t
        self.sc_id = sc_id_arr
        self.unique_ids = np.unique(self.sc_id)

        # Normalize physical properties (from columns if available)
        self._normalize_properties(len(t), df=df_all)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_timescale(self) -> None:
        """Validate the time scale
        
        Time is required to have an attached scale so that the Spacecraft
        class can work with spiceypy

        Raises:
            ValueError: the time scale must be one of 'GPS', 'TAI', 'UTC', 'ET', 'TDB'
        """        
        allowed = {'GPS', 'TAI', 'UTC', 'ET', 'TDB'}
        if self.tscale_input is None:
            raise ValueError("Time scale of the spacecraft, tscale_input, must be set")
        else:
            tscale = self.tscale_input.upper()
            if tscale not in allowed:
                raise ValueError(f"tscale_input must be one of {allowed}")
            self.tscale = tscale

    def _normalize_properties(self, N: int, df: pd.DataFrame | None = None) -> None:
        """Normalize scalar or array spacecraft properties to strict (N,) arrays."""

        # helper function to quickly process all properties
        def norm(val, col):
            if val is not None:
                arr = np.asarray(val)
                if arr.ndim == 0:
                    return np.full(N, float(arr))
                if arr.ndim == 1 and len(arr) == N:
                    return arr.astype(float)
                raise ValueError("Property must be scalar or length N")
            if df is not None and col is not None and col in df.columns:
                return df[col].to_numpy(dtype=float)
            return np.full(N, np.nan)

        # process all properties using helper function
        self.cd_arr = norm(self.cd, self.cd_col)
        self.drag_area_arr = norm(self.drag_area, self.drag_area_col)
        self.sc_mass_arr = norm(self.sc_mass, self.sc_mass_col)
        self.cr_arr = norm(self.cr, self.cr_col)
        self.srp_area_arr = norm(self.srp_area, self.srp_area_col)

    def _normalize_units(self):
        """Convert input units to SI (meters, m/s).

        Raises:
            ValueError: if unit is not 'm' or 'km'
        """
        if self.unit in ["km", "kilometer", "kilometre"]:
            self.state_ecef *= 1000.0
        elif self.unit in ["m", "meter", "metre"]:
            pass
        else:
            raise ValueError(f"Unsupported unit: {self.unit}")
        
    def _normalize_time(self):

        self.sspice_et = time_utils.spice_time(self.stime,self.tscale, 'ET')
        self.sspice_gps = time_utils.spice_time(self.stime,self.tscale, 'GPS')
        self.sc_utc = spice.et2datetime(self.sspice_et)

    def _is_sorted_asc(self, arr):
        """Check if a numpy array is sorted in ascending order."""
        if len(arr) < 2:
            return True
        # Check if all differences between consecutive elements are >= 0
        return (np.diff(arr) >= 0).all()

    def _expand_files(self,
                      state_file: str | Path | Iterable[str | Path],) -> list[Path]:
        """Expand wildcards and iterables into a sorted list of files.

        Args:
            state_file (str | Path | Iterable[str  |  Path]): Input files to search for

        Raises:
            FileNotFoundError: files not found

        Returns:
            list[Path]: List of paths to load.
        """        
        if isinstance(state_file, (str, Path)):
            paths = glob.glob(str(state_file))
        else:
            paths = []
            for f in state_file:
                paths.extend(glob.glob(str(f)))

        files = [Path(p) for p in paths]
        if not files:
            raise FileNotFoundError("No matching state files found")

        return sorted(files)

    def _load_table(self,
                    file: Path,
                    loader: str | None = None,
                    read_kwargs: dict | None = None,) -> pd.DataFrame:
        """Load file into a pandas DataFrame

        Args:
            file (Path): File to load.
            loader (str | None, optional): Loader to use, csv, hdf, or sp3. Defaults to None
            and uses file extension to infer which to use.
            read_kwargs (dict | None, optional): Keyword argument to pass to the loader 
            functions. Defaults to None.

        Raises: 
            ValueError: Raise error if loader or file extension can't be identified.

        Returns:
            pd.DataFrame: DataFrame containing Spacecraft data loaded from passed files.
        """        
        """Dispatch to the appropriate reader based on loader or file extension."""
        suffix = file.suffix.lower()
        loader = loader or suffix.lstrip(".")
        kwargs = read_kwargs or {}

        base_txt = {"csv", "txt", "text"}
        zip_txt = {"gz", "bz2", "zip", "xz", "zst",
                   "tar", "tar.gz", "tar.xz", "tar.bz2"}

        if loader in (base_txt | zip_txt):
            return pd.read_csv(file, **kwargs)
        elif loader in {"hdf", "h5"}:
            return pd.read_hdf(file, **kwargs)
        elif loader == "sp3":
            return utils.df_sp3(file, **kwargs)
        else:
            raise ValueError(f"Unsupported loader type: {loader}")

    def split_by_id(self) -> dict:
        """Split a multi-ID Spacecraft container into individual
        Spacecraft objects keyed by spacecraft ID.

        Sort time to make sure it's monotonic.

        Returns
        -------
        dict
            {spacecraft_id: Spacecraft}
        """
        spacecraft_dict = {}

        for uid in self.unique_ids:
            
            # Find the indexes that match 
            # the spacecraft it uid
            # check if they're sorted
            # if not sort 
            # sorting is done here as it reduces
            # the amount of sorting we need to do
            idx = np.where(self.sc_id == uid)[0]
            if not self._is_sorted_asc(self.sspice_gps[idx]):
                order = np.argsort(self.sspice_gps[idx])
                idx = idx[order]

            sc = Spacecraft(
                state=self.state_ecef[idx],
                time=self.stime[idx],
                sc_id_input=np.full(idx.shape[0], uid),
                tscale_input=self.tscale,
                unit_input='m',
                cd=self.cd_arr[idx],
                drag_area=self.drag_area_arr[idx],
                sc_mass=self.sc_mass_arr[idx],
                cr=self.cr_arr[idx],
                srp_area=self.srp_area_arr[idx],
            )

            spacecraft_dict[uid] = sc

        return spacecraft_dict

    # ------------------------------------------------------------------
    # Grouped normalized state view this might create overhead if we are
    # grabbing data lots but it cleans up the namespace.
    # To limit overhead the container is cached in state_data
    # -----------------------------------------------------------------
    @dataclass
    class SpacecraftState:
        state_ecef: npt.NDArray[np.float64]
        stime: pd.DatetimeIndex
        sspice_et: npt.NDArray[np.float64]
        sspice_gps: npt.NDArray[np.float64]
        sc_utc: npt.NDArray[np.float64]
        sc_id: npt.NDArray
        unique_ids: npt.NDArray
        cd: npt.NDArray
        drag_area: npt.NDArray
        sc_mass: npt.NDArray
        cr: npt.NDArray
        srp_area: npt.NDArray

        @property
        def N(self) -> int:
            return self.state_ecef.shape[0]

        @property
        def r(self) -> npt.NDArray[np.float64]:
            return self.state_ecef[:, 0:3]

        @property
        def v(self) -> npt.NDArray[np.float64]:
            return self.state_ecef[:, 3:6]

    @property
    def state_data(self) -> "Spacecraft.SpacecraftState":
        """
        Structured grouped view of canonical internal state.

        Cached after first access.
        Cache automatically invalidated whenever state is reloaded.
        """
        if not hasattr(self, "_state_data_cache"):
            self._state_data_cache = Spacecraft.SpacecraftState(
                state_ecef=self.state_ecef,
                stime=self.stime,
                sspice_et=self.sspice_et,
                sspice_gps=self.sspice_gps,
                sc_utc=self.sc_utc,
                sc_id=self.sc_id,
                unique_ids=self.unique_ids,
                cd=self.cd_arr,
                drag_area=self.drag_area_arr,
                sc_mass=self.sc_mass_arr,
                cr=self.cr_arr,
                srp_area=self.srp_area_arr,
            )
        return self._state_data_cache
    
    def _clear_cache(self) -> None:
        """Clear cached derived/grouped state containers."""
        if hasattr(self, "_state_data_cache"):
            del self._state_data_cache

    def spherical(self) -> npt.NDArray[np.float64]:
        self.state_ecef

        r = np.linalg.norm(self.state_ecef[:,0:3], axis=1)
        lat = np.arctan2(self.state_ecef[:,2],
                         np.linalg.norm(self.state_ecef[:,0:2], axis=1))
        lon = np.arctan2(self.state_ecef[:,1],self.state_ecef[:,0])

        return np.array([r,lat,lon]).transpose()
    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def N(self) -> int:
        """Number of elements in Spacecraft state/time

        Returns:
            int: Number of elements in Spacecraft state/time
        """        
        return self.state_ecef.shape[0]
    
    @property
    def n_unique_ids(self) -> int:
        """Number of unique Spacecraft loaded

        Returns:
            int: Number of unique Spacecraft loaded
        """        
        return len(list(self.unique_ids))

    def __repr__(self) -> str:
        return f"Spacecraft(N={self.N}), n_unique_ids={self.n_unique_ids}, "

