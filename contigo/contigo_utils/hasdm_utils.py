"""
hasdm_utils.py
==============
Read and query HASDM (High Accuracy Satellite Drag Model) density archive
files distributed by Space Environment Technologies / Space Weather Division.

This uses the montly files available via the SET/SWD portal:

https://spacewx.com/hasdm/

Coordinate conventions (SET/SWD v1.6.x file format):
    HTM  - altitude above EGM-96 geoid, km  (converted to m on load)
    XLAT - geocentric latitude, degrees
    XLON - geographic longitude, degrees [0, 360)
    RHO  - atmospheric density, kg/m³

Nominal grid resolution (3-hourly SET/SWD archive):
    Δlat = 10°,  Δlon = 15°,  Δalt = 25 000 m  (175 000-825 000 m)

Caveats / design decisions
--------------------------
* Grid axes are *inferred from the data* rather than hard-coded so that
  non-standard longitude offsets (e.g., the 0.8° first-longitude artefact
  observed in some SET files) are handled transparently.

* Nearest-neighbour lookup is used for all spatial dimensions.  For drag
  applications where density gradients matter, consider upgrading to
  trilinear interpolation (see ``interp_along_track`` stub at the bottom).

* The satellite time series is assumed to be *roughly* UTC and is matched
  to the nearest HASDM 3-hour epoch.  A configurable ``max_dt`` guard
  (default = half the 3-hour HASDM cadence) flags epochs where no
  satellite data is available nearby.

* Longitude is treated as a *circular* dimension so that the 0°/360°
  seam is handled correctly.
"""
from __future__ import annotations

import gzip
import io
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class HASDMGrid:
    """
    HASDM density grid parsed from a SET/SWD archive file.

    Attributes
    ----------
    density : ndarray, shape (n_t, n_lat, n_lon, n_alt)
        Atmospheric density in kg/m³.  Index order matches the axis arrays
        below so that ``density[i, j, k, l]`` corresponds to
        ``times[i], lat[j], lon[k], alt[l]``.
    times : ndarray of datetime64[ns, UTC], shape (n_t,)
        UTC epoch of each HASDM snapshot.
    lat : ndarray, shape (n_lat,)
        Geocentric latitudes in degrees, ascending.
    lon : ndarray, shape (n_lon,)
        Geographic longitudes in degrees, [0, 360), ascending.
    alt : ndarray, shape (n_alt,)
        Altitudes above EGM-96 geoid in metres (m), ascending.
    """
    density: np.ndarray
    times:   np.ndarray
    lat:     np.ndarray
    lon:     np.ndarray
    alt:     np.ndarray

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.density.shape  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"HASDMGrid(n_t={len(self.times)}, n_lat={len(self.lat)}, "
            f"n_lon={len(self.lon)}, n_alt={len(self.alt)}, "
            f"t=[{self.times[0]} … {self.times[-1]}])"
        )


class HASDMMatch(NamedTuple):
    """
    Result of matching one HASDM epoch to the satellite track.

    Attributes
    ----------
    i_time : int
        Index into ``HASDMGrid.times``.
    i_lat  : int
        Index into ``HASDMGrid.lat`` (-1 when gap > max_dt).
    i_lon  : int
        Index into ``HASDMGrid.lon`` (-1 when gap > max_dt).
    i_alt  : int
        Index into ``HASDMGrid.alt`` (-1 when gap > max_dt).
    i_sat  : int
        Index into the satellite arrays (closest sample in time).
    dt_sec : float
        Absolute time difference in seconds between the HASDM epoch and
        the matched satellite sample.
    """
    i_time: int
    i_lat:  int
    i_lon:  int
    i_alt:  int
    i_sat:  int
    dt_sec: float


# ---------------------------------------------------------------------------
# File reader — private helpers
# ---------------------------------------------------------------------------

def _parse_hasdm_file(filepath: Path) -> pd.DataFrame:
    """
    Read one HASDM archive file and return its records as a DataFrame.

    Handles plain text, gzip, and zip transparently based on the file
    suffix.  The returned DataFrame has columns:
    ``time`` (datetime64[ns] UTC tz-naive), ``lat``, ``lon``, ``alt``,
    ``rho``.

    Parameters
    ----------
    filepath : Path
        Must already be a resolved ``Path`` object.

    Raises
    ------
    ValueError
        If the file contains no valid records, or if a zip archive
        contains more than one member.
    """
    # --- resolve compression from suffix ------------------------------------
    suffix = filepath.suffix.lower()
    if suffix == ".gz":
        _opener    = lambda: gzip.open(filepath, "rt", encoding="utf-8")
        _csv_kw: dict = {"compression": "gzip"}
    elif suffix == ".zip":
        zf      = zipfile.ZipFile(filepath, "r")
        members = zf.namelist()
        if len(members) != 1:
            raise ValueError(
                f"Expected exactly 1 file inside {filepath.name}, "
                f"found {len(members)}: {members}"
            )
        _opener = lambda: io.TextIOWrapper(zf.open(members[0]), encoding="utf-8")
        _csv_kw = {"compression": "zip"}
    else:
        _opener = lambda: filepath.open("r", encoding="utf-8")
        _csv_kw = {}

    # --- count header lines (lines starting with ':' or '#') ----------------
    n_skip = 0
    with _opener() as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith(":") and not stripped.startswith("#"):
                break
            n_skip += 1

    # --- bulk read ----------------------------------------------------------
    col_names = ["datetime_str", "julian_day", "alt", "lat", "lon", "lst", "rho"]
    df = pd.read_csv(
        filepath,
        skiprows=n_skip,
        sep=r"\s+",
        header=None,
        names=col_names,
        dtype={
            "datetime_str": str,
            "julian_day":   np.float64,
            "alt":          np.float64,
            "lat":          np.float64,
            "lon":          np.float64,
            "lst":          np.float64,
            "rho":          np.float64,
        },
        **_csv_kw,
    )

    if df.empty:
        raise ValueError(f"No valid data records found in {filepath}")

    # --- parse timestamps to tz-naive UTC datetime64[ns] --------------------
    # tz_localize(None) drops the tz tag while keeping the UTC values so
    # that numpy searchsorted and comparisons work without friction.
    df["time"] = (
        pd.to_datetime(df["datetime_str"], format="%Y%m%d%H%M", utc=True)
        .dt.tz_localize(None)
    )

    # --- convert altitude from km (file native) to metres (SI) ----------
    df["alt"] = df["alt"] * 1_000.0

    # --- snap spatial coordinates to a consistent grid ---------------------
    #
    # Latitude / altitude: values are fixed across epochs but carry small
    # floating-point noise.  Round to far finer than the grid spacing to
    # collapse noise without losing precision:
    #   lat : 4 d.p.  (grid = 10°      -> tolerance 0.0001 deg, 1 : 100 000)
    #   alt : 1 d.p.  (grid = 25 000 m -> tolerance 0.1 m,      1 : 250 000)
    df["lat"] = df["lat"].round(4)
    df["alt"] = df["alt"].round(1)

    # Longitude: HASDM stores data on a fixed LOCAL SOLAR TIME (LST) grid
    # of 24 bins (one per hour, 15 deg apart).  The *geographic* longitude
    # of each LST bin drifts slowly with time as the Sun's apparent
    # position moves due to Earth's orbital motion (~0.01 deg per 3-hour
    # epoch, ~0.08 deg/day).  Over a monthly file this produces ~5 000
    # distinct geographic longitude values instead of the expected 24.
    #
    # Fix: snap each geographic longitude to the nearest multiple of 15 deg,
    # then wrap into [0, 360).  The maximum drift observed is ~3.5 deg per
    # month, well within the 7.5 deg half-spacing threshold, so no value
    # will ever snap to the wrong grid column.
    df["lon"] = ((df["lon"] / 15.0).round() * 15.0) % 360.0

    return df[["time", "lat", "lon", "alt", "rho"]]


def _build_hasdm_grid(df: pd.DataFrame) -> HASDMGrid:
    """
    Build a :class:`HASDMGrid` from a combined, pre-cleaned DataFrame.

    The DataFrame must have columns ``time``, ``lat``, ``lon``, ``alt``,
    ``rho`` and is assumed to be already deduplicated and sorted by time.

    Raises
    ------
    ValueError
        If the record count is inconsistent with the inferred grid shape.

    Warns
    -----
    UserWarning
        If any NaN cells remain in the density array after filling.
    """
    # --- infer grid axes from the data --------------------------------------
    times = np.sort(df["time"].unique())   # datetime64[ns] UTC tz-naive
    lats  = np.sort(df["lat"].unique())    # geocentric deg
    lons  = np.sort(df["lon"].unique())    # geographic deg [0, 360)
    alts  = np.sort(df["alt"].unique())    # metres above EGM-96

    n_t, n_lat, n_lon, n_alt = len(times), len(lats), len(lons), len(alts)

    # --- sanity check -------------------------------------------------------
    expected = n_t * n_lat * n_lon * n_alt
    if len(df) != expected:
        raise ValueError(
            f"Record count mismatch: found {len(df)} records, but the "
            f"inferred grid has {n_t}(t) × {n_lat}(lat) × {n_lon}(lon) × "
            f"{n_alt}(alt) = {expected} cells.  "
            "File may be truncated or contain duplicate records."
        )

    # --- vectorised index mapping -------------------------------------------
    # searchsorted is exact here because both sides come from the same
    # parsed floats — no independent coordinate arithmetic involved.
    t_idx   = np.searchsorted(times, df["time"].values)
    lat_idx = np.searchsorted(lats,  df["lat"].values)
    lon_idx = np.searchsorted(lons,  df["lon"].values)
    alt_idx = np.searchsorted(alts,  df["alt"].values)

    # --- fill density array -------------------------------------------------
    density = np.full((n_t, n_lat, n_lon, n_alt), np.nan, dtype=np.float64)
    density[t_idx, lat_idx, lon_idx, alt_idx] = df["rho"].values

    if np.any(np.isnan(density)):
        n_nan = int(np.sum(np.isnan(density)))
        warnings.warn(
            f"{n_nan} grid cell(s) remain NaN after parsing — "
            "check file for missing records.",
            UserWarning,
            stacklevel=2,
        )

    return HASDMGrid(density=density, times=times, lat=lats, lon=lons, alt=alts)


# ---------------------------------------------------------------------------
# File reader — public API
# ---------------------------------------------------------------------------

def read_hasdm(
    filepaths: str | Path | list[str | Path],
) -> HASDMGrid:
    """
    Parse one or more SET/SWD HASDM density archive files into a single
    4-D grid.

    Passing a list is the standard way to span multiple months or to
    combine a sequence of files downloaded from the SET/SWD portal.

    Supported file formats (per file, mixed lists are fine):

    * ``.txt``    - plain text
    * ``.txt.gz`` - gzip-compressed text
    * ``.gz``     - gzip-compressed text
    * ``.zip``    - zip archive containing a single ``.txt`` member

    Parameters
    ----------
    filepaths : str, Path, or list of str / Path
        One path or a list of paths to HASDM archive files.

    Returns
    -------
    HASDMGrid
        Combined density grid.  ``density`` shape:
        ``(n_t, n_lat, n_lon, n_alt)`` where ``n_t`` spans all unique
        timestamps found across all files.

    Raises
    ------
    ValueError
        * If any individual file contains no valid records.
        * If a zip archive contains more than one member.
        * If the final record count is inconsistent with the inferred
          grid shape (truncated file or residual duplicates).

    Warns
    -----
    UserWarning
        * If duplicate records are found across files — duplicates are
          dropped (first occurrence kept) before building the grid.
        * If any NaN cells remain in the density array after filling.
    """
    # --- normalise to a list of Path objects --------------------------------
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]
    paths = [Path(p) for p in filepaths]

    if not paths:
        raise ValueError("filepaths must not be empty.")

    # --- parse each file independently --------------------------------------
    # Each file gets its own header-skip and compression handling; errors
    # are re-raised with the offending filename in the message.
    frames: list[pd.DataFrame] = []
    for p in paths:
        try:
            frames.append(_parse_hasdm_file(p))
        except Exception as exc:
            raise ValueError(f"Failed to parse {p.name}: {exc}") from exc

    # --- concatenate --------------------------------------------------------
    df = pd.concat(frames, ignore_index=True)

    # --- deduplicate on (time, lat, lon, alt) --------------------------------
    # A duplicate arises when consecutive monthly files share a boundary
    # timestep, or when the same file is passed twice.
    dup_mask = df.duplicated(subset=["time", "lat", "lon", "alt"], keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        # Report in terms of epochs rather than raw rows so the count is
        # meaningful: one duplicated epoch = n_lat * n_lon * n_alt rows.
        n_lat = df["lat"].nunique()
        n_lon = df["lon"].nunique()
        n_alt = df["alt"].nunique()
        n_epochs_dup = n_dup // max(n_lat * n_lon * n_alt, 1)
        warnings.warn(
            f"Found {n_dup} duplicate records (~{n_epochs_dup} epoch(s)) "
            f"across {len(paths)} file(s) — keeping first occurrence.",
            UserWarning,
            stacklevel=2,
        )
        df = df[~dup_mask]

    # --- sort by time -------------------------------------------------------
    df = df.sort_values("time").reset_index(drop=True)

    # --- build and return the grid ------------------------------------------
    return _build_hasdm_grid(df)


# ---------------------------------------------------------------------------
# Spatial index lookup
# ---------------------------------------------------------------------------

def _circular_nearest(values: np.ndarray,
                       queries: np.ndarray,
                       period: float = 360.0) -> np.ndarray:
    """
    Vectorised nearest-neighbour on a circular axis.

    Returns the index in ``values`` nearest to each element of ``queries``
    accounting for wrap-around at ``period``.

    Parameters
    ----------
    values : ndarray, shape (n,)
        Sorted 1-D array of axis coordinates.
    queries : ndarray, shape (m,)
        Query coordinates, normalised to ``[0, period)`` internally.
    period : float
        Period of the circular dimension (360.0 for longitude).

    Returns
    -------
    ndarray of int, shape (m,)
    """
    queries = queries % period                             # (m,)
    delta = np.abs(values[:, None] - queries[None, :])    # (n, m)
    delta = np.minimum(delta, period - delta)             # wrap-around
    return np.argmin(delta, axis=0).astype(int)           # (m,)


def find_spatial_index(
    grid: HASDMGrid,
    sat_lat: float,
    sat_lon: float,
    sat_alt: float,
) -> tuple[int, int, int]:
    """
    Return the nearest-neighbour ``(i_lat, i_lon, i_alt)`` indices for a
    single satellite position.

    Parameters
    ----------
    grid : HASDMGrid
    sat_lat : float
        Geocentric latitude in degrees ``[-90, 90]``.
    sat_lon : float
        Geographic longitude in degrees (any range; normalised to
        ``[0, 360)`` internally).
    sat_alt : float
        Altitude above EGM-96 in metres (m).

    Returns
    -------
    tuple of int
        ``(i_lat, i_lon, i_alt)``

    Notes
    -----
    * Altitude is implicitly clamped to the grid boundary by ``argmin``.
      If the satellite is below 175 000 m or above 825 000 m, the nearest
      boundary altitude is returned — no exception is raised, but the
      returned density will be the boundary value, not an extrapolation.
    """
    i_lat = int(np.argmin(np.abs(grid.lat - sat_lat)))
    i_lon = int(_circular_nearest(grid.lon, np.array([sat_lon]))[0])
    i_alt = int(np.argmin(np.abs(grid.alt - sat_alt)))
    return i_lat, i_lon, i_alt


# ---------------------------------------------------------------------------
# Time-space matching: HASDM epochs ↔ satellite track
# ---------------------------------------------------------------------------

def match_satellite_to_hasdm(
    grid:      HASDMGrid,
    sat_times: np.ndarray,
    sat_lat:   np.ndarray,
    sat_lon:   np.ndarray,
    sat_alt:   np.ndarray,
    max_dt:    pd.Timedelta | None = pd.Timedelta(hours=1, minutes=30),
) -> list[HASDMMatch]:
    """
    For each HASDM epoch, find the satellite state closest in time and
    return its 4-D grid index.

    The strategy is:
    1. For each HASDM epoch (3-hourly), find ``i_sat`` — the index of the
       satellite sample nearest in time using a vectorised ``searchsorted``
       on the sorted satellite time array.
    2. Convert the matched ``(lat, lon, alt)`` to ``(i_lat, i_lon, i_alt)``
       using nearest-neighbour lookup (vectorised across all HASDM epochs).

    Parameters
    ----------
    grid : HASDMGrid
        Parsed HASDM grid.
    sat_times : ndarray of datetime64, shape (n_sat,)
        UTC timestamps of satellite state vectors.  **Must be sorted
        ascending.**  An in-place sort is performed if they are not.
    sat_lat : ndarray, shape (n_sat,)
        Geocentric satellite latitudes (degrees).
    sat_lon : ndarray, shape (n_sat,)
        Geographic satellite longitudes (degrees, any range).
    sat_alt : ndarray, shape (n_sat,)
        Satellite altitudes above EGM-96 (m).
    max_dt : pd.Timedelta or None
        Maximum tolerated time gap between a HASDM epoch and the matched
        satellite sample (default = 1 h 30 min = half the 3-h cadence).
        Matches beyond this threshold have spatial indices set to ``-1``
        and density will be ``NaN`` in the output.
        Pass ``None`` to disable.

    Returns
    -------
    list of HASDMMatch
        One entry per HASDM epoch in ``grid.times``.

    Raises
    ------
    ValueError
        If input satellite arrays differ in length.
    """
    n_sat = len(sat_times)
    if not (len(sat_lat) == len(sat_lon) == len(sat_alt) == n_sat):
        raise ValueError(
            "sat_times, sat_lat, sat_lon, and sat_alt must all have the same length."
        )

    # --- ensure satellite times are sorted (searchsorted requires this) -----
    sat_times = np.asarray(sat_times, dtype="datetime64[ns]")
    sort_order = np.argsort(sat_times)
    if not np.all(sort_order == np.arange(n_sat)):
        warnings.warn(
            "sat_times is not sorted; sorting in-place for matching.",
            UserWarning, stacklevel=2,
        )
        sat_times = sat_times[sort_order]
        sat_lat   = sat_lat[sort_order]
        sat_lon   = sat_lon[sort_order]
        sat_alt   = sat_alt[sort_order]

    sat_ns  = sat_times.astype(np.int64)   # ns since epoch
    grid_ns = grid.times.astype("datetime64[ns]").astype(np.int64)

    max_dt_ns = (
        int(max_dt.total_seconds() * 1e9)
        if max_dt is not None
        else np.iinfo(np.int64).max
    )

    # --- vectorised nearest-time matching via searchsorted ------------------
    # insert_pos[i] = where grid_ns[i] would sit in sat_ns
    insert_pos = np.searchsorted(sat_ns, grid_ns)           # (n_hasdm,)

    i_left  = np.clip(insert_pos - 1, 0, n_sat - 1)
    i_right = np.clip(insert_pos,     0, n_sat - 1)

    diff_left  = np.abs(sat_ns[i_left]  - grid_ns)
    diff_right = np.abs(sat_ns[i_right] - grid_ns)

    # pick whichever neighbour is closer
    i_sat_arr = np.where(diff_right <= diff_left, i_right, i_left)  # (n_hasdm,)
    dt_ns_arr = np.where(diff_right <= diff_left, diff_right, diff_left)

    # --- build valid/invalid mask -------------------------------------------
    valid = dt_ns_arr <= max_dt_ns   # (n_hasdm,)

    # --- vectorised spatial lookup for valid epochs -------------------------
    valid_idx = np.where(valid)[0]

    # initialise all spatial indices to -1 (sentinel for "no data")
    i_lat_arr = np.full(len(grid_ns), -1, dtype=int)
    i_lon_arr = np.full(len(grid_ns), -1, dtype=int)
    i_alt_arr = np.full(len(grid_ns), -1, dtype=int)

    if valid_idx.size > 0:
        sat_lats_v = sat_lat[i_sat_arr[valid_idx]]
        sat_lons_v = sat_lon[i_sat_arr[valid_idx]]
        sat_alts_v = sat_alt[i_sat_arr[valid_idx]]

        # latitude - standard nearest neighbour
        i_lat_arr[valid_idx] = np.argmin(
            np.abs(grid.lat[:, None] - sat_lats_v[None, :]), axis=0
        )

        # longitude - circular nearest neighbour
        i_lon_arr[valid_idx] = _circular_nearest(grid.lon, sat_lons_v)

        # altitude - standard nearest neighbour
        i_alt_arr[valid_idx] = np.argmin(
            np.abs(grid.alt[:, None] - sat_alts_v[None, :]), axis=0
        )

    # --- assemble results ---------------------------------------------------
    results: list[HASDMMatch] = [
        HASDMMatch(
            i_time=i,
            i_lat=int(i_lat_arr[i]),
            i_lon=int(i_lon_arr[i]),
            i_alt=int(i_alt_arr[i]),
            i_sat=int(i_sat_arr[i]),
            dt_sec=float(dt_ns_arr[i]) / 1e9,
        )
        for i in range(len(grid_ns))
    ]

    n_invalid = int(np.sum(~valid))
    if n_invalid:
        warnings.warn(
            f"{n_invalid} HASDM epoch(s) had no satellite data within "
            f"{max_dt}; those entries will have NaN density.",
            UserWarning,
            stacklevel=2,
        )

    return results


# ---------------------------------------------------------------------------
# Convenience extractor
# ---------------------------------------------------------------------------

def extract_along_track_density(
    grid:      HASDMGrid,
    sat_times: np.ndarray,
    sat_lat:   np.ndarray,
    sat_lon:   np.ndarray,
    sat_alt:   np.ndarray,
    max_dt:    pd.Timedelta | None = pd.Timedelta(hours=1, minutes=30),
) -> pd.DataFrame:
    """
    Extract HASDM density along a satellite track.

    Convenience wrapper around :func:`match_satellite_to_hasdm` that
    returns a tidy ``DataFrame`` ready for analysis or export.

    Parameters
    ----------
    grid : HASDMGrid
    sat_times, sat_lat, sat_lon, sat_alt :
        Satellite state (see :func:`match_satellite_to_hasdm`).
    max_dt : pd.Timedelta or None
        Maximum time gap guard.

    Returns
    -------
    pd.DataFrame
        One row per HASDM epoch with columns:

        ==================  =================================================
        ``hasdm_time``      HASDM epoch (UTC, datetime64)
        ``sat_time``        Matched satellite sample timestamp (UTC)
        ``sat_lat``         Satellite geocentric latitude (deg)
        ``sat_lon``         Satellite longitude (deg)
        ``sat_alt``         Satellite altitude above EGM-96 (m)
        ``hasdm_lat``       Nearest HASDM grid latitude (deg)
        ``hasdm_lon``       Nearest HASDM grid longitude (deg)
        ``hasdm_alt``       Nearest HASDM grid altitude (m)
        ``rho``             HASDM density (kg/m³); NaN when gap > max_dt
        ``dt_sec``          |HASDM epoch − satellite sample| in seconds
        ``i_time``          Index into grid.times
        ``i_lat``           Index into grid.lat
        ``i_lon``           Index into grid.lon
        ``i_alt``           Index into grid.alt
        ==================  =================================================
    """
    matches = match_satellite_to_hasdm(
        grid, sat_times, sat_lat, sat_lon, sat_alt, max_dt=max_dt
    )

    # ensure sat_times is a datetime64 array (may have been sorted inside the matcher)
    sat_times_ns = np.asarray(sat_times, dtype="datetime64[ns]")
    sort_order = np.argsort(sat_times_ns)
    sat_times_sorted = sat_times_ns[sort_order]
    sat_lat_sorted   = sat_lat[sort_order]
    sat_lon_sorted   = sat_lon[sort_order]
    sat_alt_sorted   = sat_alt[sort_order]

    rows = []
    for m in matches:
        hasdm_time = grid.times[m.i_time]
        sat_time   = sat_times_sorted[m.i_sat]

        if m.i_lat == -1:
            rows.append(dict(
                hasdm_time=hasdm_time,
                sat_time=sat_time,
                sat_lat=float(sat_lat_sorted[m.i_sat]),
                sat_lon=float(sat_lon_sorted[m.i_sat]),
                sat_alt=float(sat_alt_sorted[m.i_sat]),
                hasdm_lat=np.nan,
                hasdm_lon=np.nan,
                hasdm_alt=np.nan,
                rho=np.nan,
                dt_sec=m.dt_sec,
                i_time=m.i_time,
                i_lat=m.i_lat,
                i_lon=m.i_lon,
                i_alt=m.i_alt,
            ))
        else:
            rows.append(dict(
                hasdm_time=hasdm_time,
                sat_time=sat_time,
                sat_lat=float(sat_lat_sorted[m.i_sat]),
                sat_lon=float(sat_lon_sorted[m.i_sat]),
                sat_alt=float(sat_alt_sorted[m.i_sat]),
                hasdm_lat=float(grid.lat[m.i_lat]),
                hasdm_lon=float(grid.lon[m.i_lon]),
                hasdm_alt=float(grid.alt[m.i_alt]),
                rho=float(grid.density[m.i_time, m.i_lat, m.i_lon, m.i_alt]),
                dt_sec=m.dt_sec,
                i_time=m.i_time,
                i_lat=m.i_lat,
                i_lon=m.i_lon,
                i_alt=m.i_alt,
            ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Future work: trilinear interpolation stub
# ---------------------------------------------------------------------------

def interp_along_track(
    grid:    HASDMGrid,
    matches: list[HASDMMatch],
    sat_lat: np.ndarray,
    sat_lon: np.ndarray,
    sat_alt: np.ndarray,
) -> np.ndarray:
    """
    [STUB] Trilinear interpolation of HASDM density at each matched position.

    Nearest-neighbour lookup introduces a discontinuity each time the
    satellite crosses a grid-cell boundary (~500 km in lon, ~1100 km in lat
    at LEO).  For accurate drag force modelling, replace the nearest-
    neighbour result with a trilinear interpolation over the 8 surrounding
    grid corners.

    Not yet implemented — raises NotImplementedError.
    """
    raise NotImplementedError(
        "Trilinear interpolation is not yet implemented.  "
        "Use scipy.interpolate.RegularGridInterpolator as the backend:\n\n"
        "  from scipy.interpolate import RegularGridInterpolator\n"
        "  rgi = RegularGridInterpolator(\n"
        "      (grid.lat, grid.lon, grid.alt),\n"
        "      grid.density[i_time],\n"
        "      method='linear', bounds_error=False, fill_value=np.nan\n"
        "  )\n"
        "  rho = rgi(np.column_stack([lats, lons, alts]))\n"
    )
