"""
Utilities for deriving gravitational potential.

Uses numba and jit to perform faster calculations of gravitational potential.

added: 03/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

import pyshtools
import numba
import numpy as np


def read_icgem_coeff(file_path: str, encoding: str ="ISO-8859-1"):
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
    with open(file_path, "r", encoding=encoding) as potfile:
        #reader = csv.reader(potfile, delimiter=" ", skipinitialspace=True)
        for line in potfile:
            row = line.strip()
            if len(row) < 1:
                continue
            else:
                row = row.split()

            if row[0] == 'gfc':
                if 'clm' not in dir():
                    raise ValueError("max_degree header must appear before gfc data lines")
                l = int(row[1])
                m = int(row[2])
                clm[l,m] = float(row[3])
                slm[l,m] = float(row[4])
            elif row[0] == 'max_degree':
                lmax = int(row[1])
                type(lmax)
                clm = np.zeros((lmax+1,lmax+1))
                slm = np.zeros((lmax+1,lmax+1))
            elif row[0] == 'product_type':
                product = row[1]
            elif row[0] == 'radius':
                r0 = float(row[1])
            elif row[0] == 'earth_gravity_constant':
                gm = float(row[1])

    if any(v is None for v in [product, r0, gm, lmax]):
        raise ValueError(f"ICGEM file missing required header fields in {file_path}")

    return clm, slm, {'product':product,'r0':r0, 'GM':gm, 'lmax':lmax}

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
    """
    # Get normalized Legendre functions at the target latitude
    # Note: 'geodesy' normalization is required for EGM96
    theta = np.pi/2 - lat  # Convert latitude (radians) to colatitude (radians)
    # Ensure the Legendre array is explicitly a C-contiguous 2D numpy array for Numba
    p_normalized = pyshtools.legendre.PlmBar(lmax, np.cos(theta))

    # Prepare inputs for numba-jitted function
    rad_ratio = r0 / r

    m_values = np.arange(lmax + 1, dtype=np.float64)
    cos_m_lon = np.cos(m_values * lon)
    sin_m_lon = np.sin(m_values * lon)

    # Call the numba-jitted core function, passing the 2D Legendre array
    potential_sum_core = _get_potential_numba_core(
        lmax, rad_ratio, p_normalized, clm, slm, cos_m_lon, sin_m_lon)

    # Final Potential V = (GM/r) * (1 + sum)
    v = (gm / r) * (1 + potential_sum_core)
    return v

@numba.jit(nopython=True, fastmath=True)
def _get_potential_numba_core(lmax, rad_ratio, p_normalized, 
                              clm_arr, slm_arr, cos_m_lon, sin_m_lon):
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
    """
    N = len(r)
    theta = np.pi / 2 - lat                          # (N,) colatitude

    # PlmBar only accepts a scalar — one Python call per point is unavoidable,
    # but this is now the only remaining Python-level loop.
    n_plm = (lmax + 1) * (lmax + 2) // 2
    p_all = np.empty((N, n_plm), dtype=np.float64)
    for i in range(N):
        p_all[i] = pyshtools.legendre.PlmBar(lmax, np.cos(theta[i]))

    # All remaining setup is vectorised numpy — no per-point Python calls.
    rad_ratio = (r0 / r).astype(np.float64)          # (N,)
    m_values  = np.arange(lmax + 1, dtype=np.float64)
    cos_m_lon = np.cos(np.outer(lon, m_values))       # (N, lmax+1)
    sin_m_lon = np.sin(np.outer(lon, m_values))       # (N, lmax+1)

    return _get_potential_numba_batch(
        lmax, rad_ratio, p_all,
        np.ascontiguousarray(clm), np.ascontiguousarray(slm),
        cos_m_lon, sin_m_lon, gm, r0)


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _get_potential_numba_batch(lmax, rad_ratio, p_all,
                                clm_arr, slm_arr,
                                cos_m_lon, sin_m_lon, gm, r0):
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
        rr = rad_ratio[n]               # r0 / r[n]
        potential_sum = 0.0
        for l in range(2, lmax + 1):
            inner_sum = 0.0
            r_l = rr ** l
            for m in range(l + 1):
                p_ind = l * (l + 1) // 2 + m
                inner_sum += (p_all[n, p_ind]
                              * (clm_arr[l, m] * cos_m_lon[n, m]
                                 + slm_arr[l, m] * sin_m_lon[n, m]))
            potential_sum += r_l * inner_sum

        # gm/r[n] = gm * rad_ratio[n] / r0  (since rad_ratio = r0/r)
        v[n] = (gm * rr / r0) * (1.0 + potential_sum)

    return v
