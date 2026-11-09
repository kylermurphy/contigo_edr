"""Helpers shared by the tests: constants, synthetic orbits and test doubles.

Kept out of ``conftest.py`` so tests can import them by a unique module name
(``from _support import ...``); several ``conftest.py`` files may exist.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import pandas as pd
import spiceypy as spice

import contigo.config as config
from contigo.constellation import Constellation
from contigo.contigo_utils.constants import WGS84_EARTH_ANGULAR_VELOCITY, jpl_de440
from contigo.spacecraft import Spacecraft

#: Earth GM (m^3/s^2) and rotation rate (rad/s) used by the synthetic orbits.
#: OMEGA must be the value EDRDensity uses, or the Jacobi integral drifts.
GM_EARTH = jpl_de440["EARTH"]
OMEGA = WGS84_EARTH_ANGULAR_VELOCITY
R_EARTH = 6_378_137.0

# ESA POD physical properties (as in CONTIGO_Colab.ipynb)
ESA_PROPS = dict(cd=3.5, drag_area=1.1, sc_mass=430.0, cr=1.8, srp_area=1.0)


def make_circular_orbit(
    alt: float = 400e3,
    inc_deg: float = 51.6,
    raan_deg: float = 0.0,
    n: int = 600,
    dt: float = 10.0,
    t0: str = "2018-01-02 00:00:00",
) -> tuple[npt.NDArray[np.float64], pd.DatetimeIndex]:
    """Exact circular Keplerian orbit, returned as an ECEF state.

    The inertial orbit is ``r_i = a (cos u P + sin u Q)`` with ``u = n_m t``
    and mean motion ``n_m = sqrt(GM / a^3)``. The Earth-fixed state is
    ``r = R3(w t) r_i`` and ``v = R3(w t) (v_i - w x r_i)``, with ``w = OMEGA``
    about z (no precession/nutation, consistent with the EDR formula).

    Parameters
    ----------
    alt : float
        Altitude above ``R_EARTH`` in metres.
    inc_deg, raan_deg : float
        Inclination and right ascension of the ascending node in degrees.
    n : int
        Number of samples.
    dt : float
        Sample spacing in seconds.
    t0 : str
        Epoch of the first sample (labelled GPS time by the fixtures).

    Returns
    -------
    state : numpy.ndarray
        (n, 6) ECEF ``[x, y, z, vx, vy, vz]`` in m and m/s.
    time : pandas.DatetimeIndex
        (n,) sample times.
    """
    a = R_EARTH + alt
    inc, raan = np.radians(inc_deg), np.radians(raan_deg)
    p_hat = np.array([np.cos(raan), np.sin(raan), 0.0])
    q_hat = np.array(
        [-np.cos(inc) * np.sin(raan), np.cos(inc) * np.cos(raan), np.sin(inc)]
    )
    t = np.arange(n) * dt
    n_m = np.sqrt(GM_EARTH / a**3)
    u = n_m * t
    r_i = a * (np.cos(u)[:, None] * p_hat + np.sin(u)[:, None] * q_hat)
    v_i = a * n_m * (-np.sin(u)[:, None] * p_hat + np.cos(u)[:, None] * q_hat)

    w_vec = np.array([0.0, 0.0, OMEGA])
    v_rel = v_i - np.cross(w_vec, r_i)
    th = OMEGA * t
    c, s = np.cos(th), np.sin(th)

    def rot(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # R3(theta) rotates inertial into Earth-fixed coordinates
        return np.column_stack(
            [c * vec[:, 0] + s * vec[:, 1], -s * vec[:, 0] + c * vec[:, 1], vec[:, 2]]
        )

    state = np.hstack([rot(r_i), rot(v_rel)])
    time = pd.date_range(t0, periods=n, freq=pd.Timedelta(seconds=dt))
    return state, time


def load_bundled_kernels() -> None:
    """Furnish the bundled SPICE kernels and mark the NAIF download as done.

    ``SPICEEphem.load_kernels`` otherwise asks NAIF for file modification
    times on first use. Kernels are furnished with the same path strings that
    ``SPICEEphem`` builds, so it recognises them as already loaded.
    """
    loaded = {spice.kdata(i, "ALL")[0] for i in range(spice.ktotal("ALL"))}
    for fname in (config.LEAP_FILE, config.PCK_FILE, "de440s.bsp"):
        path = os.path.join(config.DATA_DIR, fname)
        if path not in loaded:
            spice.furnsh(path)
    config.state["kernel_downloaded"] = True


def make_esa_constellation(df: pd.DataFrame) -> Constellation:
    """``Constellation`` with ID ``'ESA'`` in SI from an ESA POD table in km.

    ``df`` has columns ``DateTime`` (GPS) and ``x, y, z, vx, vy, vz`` in km and
    km/s; properties are ``ESA_PROPS``.
    """
    sc = Spacecraft(
        state=df[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(),
        time=df["DateTime"],
        sc_id_input=np.full(len(df), "ESA"),
        tscale_input="GPS",
        unit_input="km",
        **ESA_PROPS,
    )
    return Constellation(spacecraft=sc)


#: (id, altitude m, inclination deg, cd) of the members of ``sc_multi``
MULTI_MEMBERS = [
    ("SAT-A", 400e3, 51.6, 2.2),
    ("SAT-B", 500e3, 87.0, 2.5),
    ("SAT-C", 300e3, 97.5, 3.0),
]


class FakeProvider:
    """Ephemeris provider double that counts calls.

    Satisfies the ``EphemerisProvider`` protocol. Body positions are simple
    deterministic functions of ET (Sun at 1 AU, Moon at 384,400 km, both in
    the equatorial plane), so results are reproducible without SPICE.
    """

    ephemeris = "de440s"
    _DIST = {"SUN": 1.495978707e11, "MOON": 3.844e8}
    _RATE = {"SUN": 2e-7, "MOON": 2.66e-6}  # rad/s, arbitrary but distinct

    def __init__(self) -> None:
        self.calls = 0
        self.requested: list[np.ndarray] = []

    def __call__(self, body, utc_time=None, gps_time=None, ephem_time=None):
        self.calls += 1
        et = np.asarray(ephem_time, dtype=float)
        self.requested.append(et)
        out = np.empty((len(body), len(et), 3))
        for i, b in enumerate(body):
            ang = self._RATE[b.upper()] * et
            d = self._DIST[b.upper()]
            out[i] = np.column_stack(
                [d * np.cos(ang), d * np.sin(ang), np.zeros_like(et)]
            )
        return out


class FakePotential:
    """Point-mass potential ``GM / r`` (positive, as ``EarthPotential`` returns)."""

    name = "FakePointMassPotential"

    def __init__(self, gm: float = GM_EARTH) -> None:
        self.gm = gm

    def potential(self, constellation: Constellation) -> dict:
        return {
            k: self.gm / np.linalg.norm(sc.state_ecef[:, :3], axis=1)
            for k, sc in constellation.spacecraft.items()
        }

    def acceleration(self, constellation, solarsys_env):
        raise NotImplementedError


class FakeForce:
    """Velocity-proportional force ``a = -k v`` (ECEF), shape (N, 3)."""

    def __init__(self, k: float = 1e-6, name: str = "FakeDrag") -> None:
        self.k = k
        self.name = name

    def acceleration(self, constellation: Constellation, solarsys_env) -> dict:
        return {
            k: -self.k * sc.state_ecef[:, 3:]
            for k, sc in constellation.spacecraft.items()
        }

    def potential(self, constellation):
        raise NotImplementedError
