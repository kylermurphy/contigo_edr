"""Regenerate the EDR regression snapshot ``tests/data/edr_snapshot.npz`` (ED-07).

Run from the repository root::

    python tests/data/make_edr_snapshot.py

The snapshot is the EDR of the ``esa_pod_small`` fixture (first 1104 rows of
``data/ESA_pod.hdf``, ECEF km and km/s, GPS time; ESA properties Cd 3.5,
A 1.1 m^2, m 430 kg; built by ``_support.make_esa_constellation``, exactly as
the ``esa_constellation`` fixture) computed by ``EDRDensity.compute_edr()`` with

- ``SolarSystemEnvironment(['SUN', 'MOON'], tolerance=0.01, SPICEEphem())``
  using the bundled kernels loaded by ``_support.load_bundled_kernels``
  (``naif0012.tls``, ``earth_latest_high_prec.bpc``, ``de440s.bsp``; no network),
- ``force_models=[ThirdBodyEnv()]`` (no SRP, no drag),
- ``potential_model=EarthPotential('EIGEN-2.gfc', lmax=50)``.

The script is deterministic: it reads only bundled data and writes one float64
array ``edr`` (m^2/s^2, 1104 values) plus the configuration as strings. Only
regenerate it when a change to the EDR output is intended, and say why in the
commit message: ED-07 is a regression test, not a validation test.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/, for _support

from _support import load_bundled_kernels, make_esa_constellation  # noqa: E402

from contigo.edr_efd import EDRDensity  # noqa: E402
from contigo.ephemeris.spice_ephem import SPICEEphem  # noqa: E402
from contigo.forces.grav_pot import EarthPotential  # noqa: E402
from contigo.forces.third_body_acc import ThirdBodyEnv  # noqa: E402
from contigo.solar_system_ephem import SolarSystemEnvironment  # noqa: E402

OUT = HERE / "edr_snapshot.npz"
CONFIG = (
    "esa_pod_small; ThirdBodyEnv(SUN, MOON; SPICEEphem de440s; tol 0.01); "
    "EarthPotential(EIGEN-2.gfc, lmax=50)"
)


def compute_snapshot() -> np.ndarray:
    """Return the ED-07 EDR (m^2/s^2) for ``esa_pod_small``."""
    df = pd.read_csv(HERE / "esa_pod_small.csv.gz", float_precision="round_trip")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    con = make_esa_constellation(df)
    env = SolarSystemEnvironment(["SUN", "MOON"], 0.01, SPICEEphem())
    edr = EDRDensity(
        con, env, [ThirdBodyEnv()], EarthPotential("EIGEN-2.gfc", lmax=50)
    ).compute_edr()
    return np.asarray(edr["ESA"], dtype=np.float64)


def main() -> None:
    load_bundled_kernels()
    edr = compute_snapshot()
    np.savez_compressed(OUT, edr=edr, config=np.array(CONFIG))
    print(f"wrote {OUT} ({edr.size} values, max |EDR| {np.abs(edr).max():.6g})")


if __name__ == "__main__":
    main()
