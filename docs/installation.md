# Installation and data

This page covers installing CONTIGO, what data it ships with, what it downloads
and where, and how to run it offline (for example on an HPC cluster).

## Install

CONTIGO needs **Python 3.10 or newer** (CI tests 3.10, 3.12, 3.13 and 3.14). It is not on
PyPI yet; install it from GitHub.

To use it:

```bash
pip install "contigo[plot] @ git+https://github.com/kylermurphy/contigo_edr.git"
```

To develop it (editable install, with test and lint tools):

```bash
git clone https://github.com/kylermurphy/contigo_edr.git
cd contigo_edr
python -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e ".[dev,plot]"
pytest                           # should pass before you change anything
```

| Extra | Adds | For |
|-------|------|-----|
| `plot` | matplotlib | `contigo.plotting` |
| `test` | pytest, pytest-cov | running the tests |
| `dev` | `test` + black, ruff, mypy, ipykernel | development |
| `docs` | mkdocs and plugins | a docs site (later) |

**Java:** Orekit is a core dependency and runs in a Java virtual machine. You do not need to
install Java: the `jdk4py` dependency ships a JDK, and `orekit-jpype` starts it. The JVM only
starts when you first use an Orekit feature (`OrekitEphem`, `SRPOrekitCB`, `jb08_utils`), never
at import.

**Google Colab:** the tutorials in `notebooks/tutorials/` have an "Open in Colab" badge and a
first cell that clones the repository and installs it.

## Environment variables

CONTIGO reads two optional environment variables. Both are read each time they are needed,
not once at import, so setting them in Python after `import contigo` also works.

| Variable | Effect |
|----------|--------|
| `CONTIGO_CACHE_DIR` | Folder for downloaded data, in place of the default cache folder (below) |
| `CONTIGO_OFFLINE` | `1` (or `true`, `yes`, `on`) stops all network requests (see [Offline use and HPC](#offline-use-and-hpc)) |

Set them for one session:

```bash
export CONTIGO_OFFLINE=1                        # Linux / macOS (bash, zsh)
$env:CONTIGO_OFFLINE = "1"                      # Windows PowerShell
set CONTIGO_OFFLINE=1                           # Windows cmd
```

or in Python, before the code that downloads:

```python
import os
os.environ["CONTIGO_CACHE_DIR"] = "/scratch/me/contigo-cache"
os.environ["CONTIGO_OFFLINE"] = "1"
```

To make them permanent, add the `export` line to `~/.bashrc` (or `~/.zshrc`), or on Windows
use **Settings > System > About > Advanced system settings > Environment Variables** (or
`setx CONTIGO_OFFLINE 1`, which applies to new terminals). On a cluster, put the `export`
lines in your job script.

## Bundled data

These files are installed with the package (`contigo/data/`, read-only). Find them with
`contigo.config.data_path()`.

| File | What it is |
|------|------------|
| `naif0012.tls` | SPICE leap-second kernel (all time conversions) |
| `de440s.bsp` | JPL planetary ephemeris (`SPICEEphem`) |
| `earth_latest_high_prec.bpc` | High-precision Earth orientation (ITRF93), used by `coords.ecef_to_inertial`. Covers 2000-01-01 to **2026-05-30** |
| `pck00011.tpc` | Text PCK with Earth's IAU rotation model (the low-precision `IAU_EARTH` fallback) |
| `EIGEN-2.gfc` | Gravity field, degree 140 (the `EarthPotential` default) |
| `CHAMP_POD_20050816_20050822.hdf` | One week of CHAMP orbit data for the tutorials |
| `contigo/java/orekit_utils-1.0.0.jar` | CONTIGO's Java helpers for Orekit |

The repository's top-level `data/` folder (`ESA_pod.hdf`, `ore_d.hdf`, ...) is example and
reference data for the tutorials. It is **not** part of the installed package.

## Downloaded data and the cache

Anything CONTIGO downloads goes to a per-user cache, never into the package:

| OS | Default cache folder |
|----|----------------------|
| Windows | `C:\Users\<you>\AppData\Local\contigo\contigo\Cache` |
| macOS | `~/Library/Caches/contigo` |
| Linux | `~/.cache/contigo` (or `$XDG_CACHE_HOME/contigo`) |

Set `CONTIGO_CACHE_DIR` to use another folder (for example a project or scratch disk).
`contigo.config.cache_dir()` returns the folder in use.

| What | When | Size | Subfolder |
|------|------|------|-----------|
| Orekit data (`orekit_data_<sha8>.zip`) | First Orekit use. Later starts reuse it; a newer version is fetched when orekit-data publishes one | ~21 MB | `orekit/` |
| `EGM2008.gfc`, `EIGEN-6C4.gfc` | First `EarthPotential(pot_file="EGM2008.gfc")` (or EIGEN-6C4). Checked against a size and SHA-256 | 250 MB, 180 MB | `gravity/` |
| SPICE kernels | When NAIF publishes a newer copy of a bundled kernel (`SPICEEphem`, once per session), or when `ecef_to_inertial` needs Earth orientation after the bundled kernel ends | 5 to 30 MB | `spice/` |

Every download is written to a temporary `.part` file and only moved into place when it is
complete, so an interrupted download never leaves a broken file behind.

**Where files are looked up:** for a bare file name (such as `"EGM2008.gfc"`) CONTIGO looks in
the current folder, then the cache, then the bundled data. A path with a folder part
(`"/data/fields/my_model.gfc"`) is used as given and never triggers a download.

## Offline use and HPC

Set `CONTIGO_OFFLINE=1` and CONTIGO makes no network requests:

- downloads raise `contigo.config.OfflineError` instead;
- the NAIF "is there a newer kernel?" check is skipped;
- Orekit uses the newest Orekit data zip in the cache;
- a missing gravity model raises `PotentialFileNotFound`, naming the URL and the folder to put
  it in;
- `ecef_to_inertial` for dates after the bundled Earth orientation kernel warns and falls back
  to `IAU_EARTH` (see [coordinates.md](coordinates.md)).

On a cluster whose compute nodes have no internet, fill the cache on a login node first:

```bash
export CONTIGO_CACHE_DIR=/scratch/$USER/contigo-cache   # shared by login and compute nodes
python - <<'EOF'
from contigo.forces.grav_utils import fetch_icgem
from contigo.contigo_utils.orekit_utils import resolve_orekit_data
from contigo.contigo_utils.kernels import check_pck
fetch_icgem("EGM2008.gfc")           # only the models you use
resolve_orekit_data()                # Orekit data zip
check_pck([8.4e8])                   # newest Earth orientation kernel (any recent ET)
EOF
```

Then run jobs with `CONTIGO_OFFLINE=1` and the same `CONTIGO_CACHE_DIR`.

## Logging

CONTIGO never configures logging. Warnings still reach stderr; to see progress messages
(downloads, kernels loaded) add this to your script:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Troubleshooting

| Message | Cause and fix |
|---------|---------------|
| `OrekitDataNotFoundError` | No network and no Orekit data zip in the cache. Go online once, or copy an `orekit_data_<sha8>.zip` into `cache_dir("orekit")` |
| `PotentialFileNotFound` | A gravity model is missing and could not be downloaded (offline). Run `fetch_icgem(name)` online, or copy the file into `cache_dir("gravity")` |
| `RuntimeWarning: No Earth orientation data (ITRF93) ...` | Your dates are after 2026-05-30 and no newer kernel could be downloaded. Positions are off by ~10 km. Go online once, or see [coordinates.md](coordinates.md) |
| `start_orekit()` warns that the JVM was started elsewhere | Something else started Java before CONTIGO. Import and use CONTIGO's Orekit features first in a fresh session |
