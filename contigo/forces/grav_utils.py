"""
Utilities for deriving gravitational potential.

Uses numba and jit to perform faster calculations of gravitational potential.

Also holds the registry of ICGEM gravity-field models that are downloaded on
demand (:data:`ICGEM_MODELS`, :func:`fetch_icgem`, :func:`resolve_pot_file`).

added: 03/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import NamedTuple

import numba
import numpy as np
import pyshtools

import contigo.config as config
from contigo.contigo_utils import utils

logger = logging.getLogger(__name__)


class IcgemModel(NamedTuple):
    """An ICGEM gravity-field model that can be downloaded on demand.

    ``size`` (bytes) and ``sha256`` are checked after download when not None.
    ``lmax`` must equal the ``max_degree`` header of the file.
    """

    url: str
    size: int | None
    sha256: str | None
    lmax: int


_ICGEM = "https://icgem.gfz.de/getmodel/gfc/"

#: Downloadable ICGEM models, keyed by file name. EIGEN-2.gfc is also bundled
#: with the package; the registry entry is used by the network tests.
ICGEM_MODELS: dict[str, IcgemModel] = {
    "EGM2008.gfc": IcgemModel(
        url=_ICGEM
        + "c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340"
        + "/EGM2008.gfc",
        size=252140870,
        sha256="ab5b524da073e63b5bdceb7ca47a0de07a26dd44a1c5798f39fc98dc80af70fd",
        lmax=2190,
    ),
    "EIGEN-6C4.gfc": IcgemModel(
        url=_ICGEM
        + "7fd8fe44aa1518cd79ca84300aef4b41ddb2364aef9e82b7cdaabdb60a9053f1"
        + "/EIGEN-6C4.gfc",
        size=177702723,
        sha256="2ac274a66a25fb25bdddcec7074669867939345ad003bbc1ece3967450d48dc1",
        lmax=2190,
    ),
    "EIGEN-2.gfc": IcgemModel(
        url=_ICGEM
        + "4dbcdba7387ea4621f5306a55e91f82e346242791f23a4912b18fc40effdca4e"
        + "/EIGEN-2.gfc",
        size=None,
        sha256=None,
        lmax=140,
    ),
}


class PotentialFileNotFound(FileNotFoundError, ValueError):
    """A gravity-field (potential) file is missing and cannot be downloaded.

    Subclasses ``ValueError`` so existing ``except ValueError`` code keeps
    working.
    """


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _read_icgem_header(path: Path, encoding: str = "ISO-8859-1") -> dict:
    """Return the ICGEM header keywords (up to ``end_of_head``) of a file."""
    head: dict[str, str] = {}
    with open(path, "r", encoding=encoding) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "end_of_head":
                break
            head[parts[0]] = parts[1] if len(parts) > 1 else ""
    return head


def _verify_icgem(path: Path, model: IcgemModel, name: str) -> None:
    """Raise IOError if ``path`` does not match the registry entry."""
    if model.size is not None:
        size = path.stat().st_size
        if size != model.size:
            raise IOError(
                f"Downloaded {name} has {size} bytes, expected {model.size} "
                f"(from {model.url})"
            )
    if model.sha256 is not None:
        digest = _sha256(path)
        if digest != model.sha256:
            raise IOError(
                f"Downloaded {name} has sha256 {digest}, expected {model.sha256} "
                f"(from {model.url})"
            )
    head = _read_icgem_header(path)
    if "modelname" not in head:
        raise IOError(f"Downloaded {name} has no 'modelname' header (from {model.url})")
    try:
        lmax = int(head.get("max_degree", ""))
    except ValueError:
        lmax = None
    if lmax != model.lmax:
        raise IOError(
            f"Downloaded {name} has max_degree {head.get('max_degree')}, expected "
            f"{model.lmax} (from {model.url})"
        )


def fetch_icgem(name: str, dest_dir: str | os.PathLike | None = None) -> Path:
    """Download a registered ICGEM gravity-field file and return its path.

    Use it to prefetch a model (for example on an HPC login node before
    running offline on compute nodes). The file is written to
    ``dest_dir`` (default ``config.cache_dir("gravity")``) via a temporary
    ``<name>.<pid>.part`` file, verified (size and sha256 when known in
    :data:`ICGEM_MODELS`, ``max_degree`` and ``modelname`` headers) and then
    moved into place atomically. An existing file is overwritten.

    Parameters
    ----------
    name : str
        A key of :data:`ICGEM_MODELS`, e.g. ``"EGM2008.gfc"``.
    dest_dir : str or path-like, optional
        Target directory, by default the per-user gravity cache.

    Returns
    -------
    Path
        Path of the downloaded file.

    Raises
    ------
    PotentialFileNotFound
        If ``CONTIGO_OFFLINE`` is set (nothing is downloaded).
    KeyError
        If ``name`` is not in the registry.
    IOError
        If the downloaded file fails verification (it is deleted).
    """
    model = ICGEM_MODELS[name]
    target_dir = (
        Path(dest_dir).expanduser()
        if dest_dir is not None
        else config.cache_dir("gravity", create=False)
    )
    if config.is_offline():
        raise PotentialFileNotFound(
            f"Potential file {name} cannot be found and CONTIGO_OFFLINE is set: "
            f"it would be downloaded from {model.url} into {target_dir}. "
            "unset CONTIGO_OFFLINE or pass an absolute path to an existing file."
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    final = target_dir / name
    part = target_dir / f"{name}.{os.getpid()}.part"
    logger.info("Downloading %s from %s to %s", name, model.url, final)
    try:
        utils.dl_file(model.url, str(part))
        _verify_icgem(part, model, name)
        os.replace(part, final)
    finally:
        if part.exists():
            part.unlink()
    return final


def resolve_pot_file(name: str | os.PathLike) -> Path:
    """Find a potential file, downloading registered ICGEM models if needed.

    1. :func:`contigo.config.find_data_file` (path as given; a bare name in
       the current directory, the gravity cache, then the bundled data).
    2. A bare file name that is a key of :data:`ICGEM_MODELS` is downloaded
       with :func:`fetch_icgem`.
    3. Otherwise :class:`PotentialFileNotFound` is raised. A missing path with
       a directory part never triggers a download.
    """
    found = config.find_data_file(name, "gravity")
    if found is not None:
        return found
    p = Path(name)
    if len(p.parts) == 1 and not p.is_absolute() and p.name in ICGEM_MODELS:
        return fetch_icgem(p.name)
    if len(p.parts) == 1 and not p.is_absolute():
        searched = [
            str(Path.cwd() / p.name),
            str(config.cache_dir("gravity", create=False) / p.name),
            str(Path(config.DATA_DIR) / p.name),
        ]
    else:
        searched = [str(p.expanduser())]
    raise PotentialFileNotFound(
        f"Potential file {name} cannot be found; searched: {', '.join(searched)}. "
        f"Downloadable models: {', '.join(sorted(ICGEM_MODELS))}."
    )


def read_icgem_coeff(file_path: str, encoding: str = "ISO-8859-1"):
    """Read the ICGEM gravitational field model coefficients.

    Parameters
    ----------
    file_path : str or path like object
        Path to ICGEM global earth data model coeffecients
    encoding : str, optional
        Encoding for reading the files, by default "ISO-8859-1".
        Some files use special characters so a more inclusive encoding is needed.

    Returns
    -------
    clm : 2-D np.array of shape [l,m] holding the C coefficients for the spherical
        harmonics
    slm : 2-D np.array of shape [l,m] holding the S coefficients for the spherical
        harmonics
    dict
        Metadata from the file: GM (m^3/s^2), r0 (m), lmax, product type.

    Reference
    ---------
    https://icgem.gfz.de/
    """
    product = r0 = gm = lmax = None
    clm = slm = None
    with open(file_path, "r", encoding=encoding) as potfile:
        # reader = csv.reader(potfile, delimiter=" ", skipinitialspace=True)
        for line in potfile:
            row = line.strip()
            if len(row) < 1:
                continue
            else:
                row = row.split()

            if row[0] == "gfc":
                if clm is None:
                    raise ValueError(
                        "max_degree header must appear before gfc data lines"
                    )
                l = int(row[1])
                m = int(row[2])
                clm[l, m] = float(
                    row[3].upper().replace("D", "e")
                    if "D" in row[3].upper()
                    else row[3]
                )
                slm[l, m] = float(
                    row[4].upper().replace("D", "e")
                    if "D" in row[4].upper()
                    else row[4]
                )
            elif row[0] == "max_degree":
                lmax = int(row[1])
                clm = np.zeros((lmax + 1, lmax + 1))
                slm = np.zeros((lmax + 1, lmax + 1))
            elif row[0] == "product_type":
                product = row[1]
            elif row[0] == "radius":
                r0 = float(row[1])
            elif row[0] == "earth_gravity_constant":
                gm = float(row[1])

    if any(v is None for v in [product, r0, gm, lmax]):
        raise ValueError(f"ICGEM file missing required header fields in {file_path}")

    return clm, slm, {"product": product, "r0": r0, "GM": gm, "lmax": lmax}


def _check_lmax(clm, slm, lmax):
    """Raise ValueError if lmax exceeds the coefficient arrays.

    The Numba kernels have no bounds checks, so an lmax above the size of
    the coefficient arrays would read past the end of them.
    """
    clm = np.asarray(clm)
    slm = np.asarray(slm)
    if clm.ndim != 2 or clm.shape != slm.shape:
        raise ValueError(
            f"clm and slm must be 2-D arrays of the same shape, got {clm.shape} "
            f"and {slm.shape}"
        )
    max_deg = min(clm.shape) - 1
    if lmax < 0 or lmax > max_deg:
        raise ValueError(
            f"lmax={lmax} is outside the coefficient arrays (maximum degree "
            f"{max_deg})"
        )


def get_potential(r, lat, lon, clm, slm, gm, r0, lmax=50):
    """Derive Gravitational Potential

    Derive all values needed for potential derivation then pass them
    to _get_potential_numba_core which uses Numba and JIT to perform
    the derivation faster, namely the sum of the double for loop.

    The units of r, gm, and r0 should be consistent. For example if r is
    in meters then r0 should be in meters and GM in m^3/s^2.

    Parameters
    ----------
    r : float
        Radial location (ECEF - length)
    lat : float
        Latitude location (ECEF - radians)
    lon : float
        Longitude location (ECEF - radians)
    clm : 2D array
        C_lm coeffecients.
    slm : _type_
        S_lm coeffecients.
    gm :float
        Gravatational Potential Constant of object (length^3/s^2)
    r0 : float
        Radius of object (length)
    lmax : int, optional
        Max degree/order for potential derivation, by default 50

    Returns
    -------
    float
        Gravitational Potential (length^2/s^2)

    Raises
    ------
    ValueError
        If lmax exceeds the degree of the coefficient arrays, or clm and slm
        have different shapes.
    """
    _check_lmax(clm, slm, lmax)
    # Get normalized Legendre functions at the target latitude
    # Note: 'geodesy' normalization is required for EGM96
    theta = np.pi / 2 - lat  # Convert latitude (radians) to colatitude (radians)
    # Ensure the Legendre array is explicitly a C-contiguous 2D numpy array for Numba
    p_normalized = pyshtools.legendre.PlmBar(lmax, np.cos(theta))

    # Prepare inputs for numba-jitted function
    rad_ratio = r0 / r

    m_values = np.arange(lmax + 1, dtype=np.float64)
    cos_m_lon = np.cos(m_values * lon)
    sin_m_lon = np.sin(m_values * lon)

    # Call the numba-jitted core function, passing the 2D Legendre array
    potential_sum_core = _get_potential_numba_core(
        lmax, rad_ratio, p_normalized, clm, slm, cos_m_lon, sin_m_lon
    )

    # Final Potential V = (GM/r) * (1 + sum)
    v = (gm / r) * (1 + potential_sum_core)
    return v


@numba.jit(nopython=True, fastmath=True)
def _get_potential_numba_core(
    lmax, rad_ratio, p_normalized, clm_arr, slm_arr, cos_m_lon, sin_m_lon
):
    """Numba/JIT function to perform potential summation

    Parameters
    ----------
    lmax : int
        Max degree/order for potential derivation, by default 50
    rad_ratio : float
        Ratio between position and body radius.
    p_normalized : float
        Normalized Legendre coefficients
    clm_arr : 2D array float
        _C_lm coeffecients.
    slm_arr : 2D array float
        S_lm coeffecients.
    cos_m_lon : float
        cos(mlon)
    sin_m_lon : float
        sin(mlon)

    Returns
    -------
    float
        Potential double for loop sum
    """
    potential_sum = 0.0

    for l in range(2, lmax + 1):
        inner_sum = 0.0
        r_l = rad_ratio**l
        for m in range(l + 1):
            c_lm = float(clm_arr[l, m])
            s_lm = float(slm_arr[l, m])

            current_cos_mlon = float(cos_m_lon[m])
            current_sin_mlon = float(sin_m_lon[m])

            p_ind = int(l * (l + 1) / 2 + m)

            # Access the Legendre polynomial from the 1D array
            p_lm = float(p_normalized[p_ind])

            inner_sum += p_lm * (c_lm * current_cos_mlon + s_lm * current_sin_mlon)

        potential_sum += r_l * inner_sum
    return potential_sum


def get_potential_batch(r, lat, lon, clm, slm, gm, r0, lmax=50):
    """Derive gravitational potential for N positions in a single Numba call.

    Batched version of get_potential. Pre-computes Legendre polynomials and
    trigonometric terms for all N positions, then dispatches to a single
    parallel Numba call rather than N serial calls.

    Parameters
    ----------
    r : np.ndarray (N,)
        Radial locations (m).
    lat : np.ndarray (N,)
        Geocentric latitudes (radians).
    lon : np.ndarray (N,)
        Longitudes (radians).
    clm : 2D array (lmax+1, lmax+1)
        C_lm coefficients.
    slm : 2D array (lmax+1, lmax+1)
        S_lm coefficients.
    gm : float
        Gravitational constant (m^3/s^2).
    r0 : float
        Reference radius (m).
    lmax : int, optional
        Max degree/order, by default 50.

    Returns
    -------
    np.ndarray (N,)
        Gravitational potential at each position (m^2/s^2).

    Raises
    ------
    ValueError
        If lmax exceeds the degree of the coefficient arrays, or clm and slm
        have different shapes.
    """
    _check_lmax(clm, slm, lmax)
    N = len(r)
    theta = np.pi / 2 - lat  # (N,) colatitude

    # PlmBar only accepts a scalar — one Python call per point is unavoidable,
    # but this is now the only remaining Python-level loop.
    n_plm = (lmax + 1) * (lmax + 2) // 2
    p_all = np.empty((N, n_plm), dtype=np.float64)
    for i in range(N):
        p_all[i] = pyshtools.legendre.PlmBar(lmax, np.cos(theta[i]))

    # All remaining setup is vectorised numpy — no per-point Python calls.
    rad_ratio = (r0 / r).astype(np.float64)  # (N,)
    m_values = np.arange(lmax + 1, dtype=np.float64)
    cos_m_lon = np.cos(np.outer(lon, m_values))  # (N, lmax+1)
    sin_m_lon = np.sin(np.outer(lon, m_values))  # (N, lmax+1)

    return _get_potential_numba_batch(
        lmax,
        rad_ratio,
        p_all,
        np.ascontiguousarray(clm),
        np.ascontiguousarray(slm),
        cos_m_lon,
        sin_m_lon,
        gm,
        r0,
    )


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _get_potential_numba_batch(
    lmax, rad_ratio, p_all, clm_arr, slm_arr, cos_m_lon, sin_m_lon, gm, r0
):
    """Parallel Numba kernel: gravitational potential for N positions.

    Parameters
    ----------
    lmax : int
        Max degree/order.
    rad_ratio : np.ndarray (N,)
        r0/r for each position.
    p_all : np.ndarray (N, n_plm)
        Normalized Legendre polynomials for all positions.
    clm_arr : 2D array (lmax+1, lmax+1)
        C_lm coefficients.
    slm_arr : 2D array (lmax+1, lmax+1)
        S_lm coefficients.
    cos_m_lon : np.ndarray (N, lmax+1)
        cos(m*lon) for all positions.
    sin_m_lon : np.ndarray (N, lmax+1)
        sin(m*lon) for all positions.
    gm : float
        Gravitational constant (m^3/s^2).
    r0 : float
        Reference radius (m).

    Returns
    -------
    np.ndarray (N,)
        Gravitational potential at each position (m^2/s^2).
    """
    N = rad_ratio.shape[0]
    v = np.empty(N, dtype=np.float64)

    for n in numba.prange(N):
        rr = rad_ratio[n]  # r0 / r[n]
        potential_sum = 0.0
        for l in range(2, lmax + 1):
            inner_sum = 0.0
            r_l = rr**l
            for m in range(l + 1):
                p_ind = l * (l + 1) // 2 + m
                inner_sum += p_all[n, p_ind] * (
                    clm_arr[l, m] * cos_m_lon[n, m] + slm_arr[l, m] * sin_m_lon[n, m]
                )
            potential_sum += r_l * inner_sum

        # gm/r[n] = gm * rad_ratio[n] / r0  (since rad_ratio = r0/r)
        v[n] = (gm * rr / r0) * (1.0 + potential_sum)

    return v
