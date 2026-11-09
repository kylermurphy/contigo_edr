"""Regenerate the small reference fixtures in tests/data/ from the full files.

Run from the repository root::

    python tests/data/make_fixtures.py

Inputs (in the repository, not shipped with the package):

- ``data/ESA_pod.hdf``: precise orbit (POD) of one LEO spacecraft, 10 s cadence,
  GPS time. Positions in **km**, velocities in **km/s** (ECEF); GMAT SRP
  acceleration ``srp_*`` in km/s^2.
- ``data/ore_d.hdf``: Orekit results for the same epochs, in **SI**: Earth
  potential ``earth_gp`` (m^2/s^2), Sun/Moon ECEF positions ``ecef_*n_p*`` (m)
  and third-body accelerations ``ecef_*n_a*`` (m/s^2), ECI state ``eci_*`` (m,
  m/s), EDR ``edr`` (m^2/s^2) and EFD denominator ``denom``. Note: its ``ecef_*``
  state columns are in km like ESA_pod, and one is named ``ecef_vx,``.

The first ``N_ROWS`` rows (about two orbits) are written as gzipped CSV with
full float precision.

added: 09/11/2026 Claude Code
"""

from pathlib import Path

import pandas as pd

N_ROWS = 1104  # ~2 orbits at 10 s
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

ESA_COLS = [
    "DateTime",
    "sat",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "srp_x",
    "srp_y",
    "srp_z",
]
ORE_COLS = (
    ["DateTime", "earth_gp", "edr", "denom"]
    + [f"ecef_{b}n_{q}{c}" for b in "sm" for q in "pa" for c in "xyz"]
    + [f"eci_{c}" for c in ("x", "y", "z", "vx", "vy", "vz")]
)


def main() -> None:
    esa = pd.read_hdf(ROOT / "data" / "ESA_pod.hdf").iloc[:N_ROWS][ESA_COLS]
    ore = pd.read_hdf(ROOT / "data" / "ore_d.hdf").iloc[:N_ROWS][ORE_COLS]
    if not (esa["DateTime"].to_numpy() == ore["DateTime"].to_numpy()).all():
        raise RuntimeError("ESA_pod and ore_d epochs are not aligned")
    esa.to_csv(OUT / "esa_pod_small.csv.gz", index=False, float_format="%.17g")
    ore.to_csv(OUT / "orekit_ref_small.csv.gz", index=False, float_format="%.17g")
    print(f"wrote {N_ROWS} rows to {OUT}")


if __name__ == "__main__":
    main()
