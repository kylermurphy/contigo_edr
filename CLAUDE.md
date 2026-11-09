# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

CONTIGO is a Python library for computing Energy Dissipation Rate (EDR) and Effective Density (EFD) of Earth-orbiting spacecraft. It is organized as a layered, modular framework where new force models can be added via a Protocol interface.

## Development Setup

Install in editable mode with dev extras:
```bash
pip install -e ".[dev]"
```

For Orekit support:
```bash
pip install -e ".[orekit]"
```

## Common Commands

```bash
# Format
black contigo/

# Lint
ruff check contigo/

# Type check
mypy contigo/

# Run tests
pytest

# Run a single test file
pytest tests/test_spacecraft.py -v
```

## Java Backend (Orekit utilities)

The precompiled JAR at `java_src/target/orekit_utils-1.0.0.jar` provides high-performance batch helpers for Orekit (`EphemerisBatchHelper.java`, `SRPCannonballBatchHelper.java`). If you change the Java source, rebuild with Maven:

```bash
cd java_src && mvn clean package
```

The JAR is auto-loaded by `orekit_utils.start_orekit()` from that path, so the JAR must exist before Orekit features can be used.

## Architecture

The framework has four layers that compose top-to-bottom:

### Data Layer
- **`Spacecraft`** (`spacecraft.py`) — loads orbit state from arrays, HDF, CSV/text, or SP3 files. Normalizes everything to SI (m, m/s) ECEF and generates three internal time axes: `sspice_et` (SPICE ephemeris time), `sspice_gps` (GPS seconds), and `sc_utc` (UTC datetimes). Physical properties (mass, Cd, Cr, areas) are broadcast to `(N,)` arrays regardless of whether a scalar or array was provided.
- **`Constellation`** (`constellation.py`) — accepts the same kwargs as `Spacecraft` or a list of pre-built `Spacecraft` objects. Internally calls `Spacecraft.split_by_id()` to produce a `dict[spacecraft_id, Spacecraft]`. All downstream code works on this dict.

### Environment Layer
- **`EphemerisProvider`** (`ephemeris/base.py`) — a `Protocol` (`__call__`) that any ephemeris backend must satisfy. Returns `(n_bodies, N, 3)` position arrays.
  - **`SPICEEphem`** — uses `spiceypy`. Loads SPICE kernels on init; kernels are cached in `contigo/data/`.
  - **`OrekitEphem`** — uses `orekit_jpype` (JVM). Defers all Java imports until after the JVM starts. Uses `EphemerisBatchHelper` from the JAR.
- **`SolarSystemEnvironment`** (`solar_system_ephem.py`) — wraps an `EphemerisProvider` with an O(1) time-quantization cache. Times are binned by `np.round(t / tolerance).astype(int)`. Only missing bins trigger a provider call. `EDRDensity` pre-loads all constellation epochs during `__init__`.

### Physics Layer
- **`ForceModel`** (`forces/base.py`) — a `Protocol` with two methods:
  - `acceleration(constellation, solarsys_env) → dict[sc_id, ndarray]` — shape `(N,3)` or `(n_bodies, N, 3)` for multi-body forces
  - `potential(constellation) → dict[sc_id, ndarray]` — shape `(N,)`, used only for the gravity potential

  Implementing this protocol is all that's needed to add a new force to the EDR computation. Concrete implementations: `ThirdBodyEnv`, `SRPAcc` (GMAT or Orekit cannonball), `EarthPotential` (pyshtools + ICGEM `.gfc` gravity field files).

### Computation Layer
- **`EDRDensity`** (`edr_efd.py`) — the core engine. Accepts a `Constellation`, a `SolarSystemEnvironment`, a list of `ForceModel` implementations, and a `potential_model`. Exposes:
  - `compute_edr()` → `dict[sc_id, ndarray]` — integrates eq. A18/A19 of [doi:10.1029/2024EA003898](https://doi.org/10.1029/2024EA003898)
  - `compute_denom()` → `dict[sc_id, ndarray]` — denominator of eq. A20
  - `compute_den(window, smth_edr, smth_den)` → `dict[sc_id, DataFrame]` — full effective density pipeline

## Global State

`contigo/config.py` holds a mutable `state` dict that tracks loaded external resources:
```python
state = {
    'orekit_loaded': False,   # True after JVM started
    'gmat_loaded': False,     # True after GMAT API loaded
    'gmatpy': None,           # GMAT Python API module reference
    'pot_coef_loaded': False, # True after gravity coefficients loaded
    'pot_file': None,         # Path to the loaded .gfc file
    'pot_clm': None, 'pot_slm': None, 'pot_r0': None, 'pot_GM': None,
}
```
Check `config.state['orekit_loaded']` before using any Orekit feature. `OrekitEphem.__init__` and Orekit SRP classes call `orekit_utils.start_orekit()` automatically if not yet started.

## Key Conventions

- All internal position/velocity is **ECEF, SI** (meters, m/s). The `unit_input='km'` parameter handles conversion on load.
- Time is always stored as three parallel arrays: `sspice_et`, `sspice_gps`, `sc_utc`. Use GPS seconds as the sort/dedup key throughout.
- `Constellation` pre-computes a union time axis over all member spacecraft — this is what `SolarSystemEnvironment._load_times()` uses to warm the cache in one batch.
- Force model `acceleration()` results are summed by `EDRDensity.compute_edr()` via `cumulative_simpson` over GPS seconds (shifted to start at zero). A 3-D `(n_bodies, N, 3)` result is summed over the body axis before integration.
- Gravity field data (`.gfc` files) for `EarthPotential` come from [ICGEM](https://icgem.gfz.de/tom_longtime); EIGEN-2 is bundled at `contigo/data/EIGEN-2.gfc`.
