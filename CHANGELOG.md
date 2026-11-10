# Changelog

Notable changes to CONTIGO. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, a minor version may break things).

## [0.2.0] - Unreleased

The "Phase 0" release: packaging, tests and CI, data downloads into a per-user cache,
coordinate transforms, and many bug fixes.

### Added

- **Coordinates** (`contigo_utils/coords.py`): `ecef_to_geodetic`, `geodetic_to_ecef`
  (WGS-84, vectorised), `ecef_to_inertial` and `inertial_to_ecef` (SPICE ITRF93 to J2000, for
  positions or full states). Constants `WGS84_A`, `WGS84_F`, `WGS84_E2`. See
  [docs/coordinates.md](docs/coordinates.md).
- **Earth orientation kernels** (`kernels.check_pck`): checks that the loaded kernel covers
  the requested dates, tries one update from NAIF per session, otherwise warns and falls back
  to `IAU_EARTH` (or raises with `fallback="raise"`). Bundled `pck00011.tpc` for the fallback.
- **SPICE kernel helpers** (`contigo_utils/kernels.py`): `naif_url`, `ensure_kernel`,
  `furnsh_kernel`, shared by `SPICEEphem` and `check_lpsk`.
- **Per-user cache** for downloads (`config.cache_dir()`, override with `CONTIGO_CACHE_DIR`)
  and `config.find_data_file()`.
- **Offline switch** `CONTIGO_OFFLINE=1`: no network requests; local files only.
- **Gravity models on demand**: `EGM2008.gfc` and `EIGEN-6C4.gfc` are downloaded from ICGEM on
  first use and checked against a size and SHA-256 (`grav_utils.ICGEM_MODELS`,
  `fetch_icgem()` to prefetch, `PotentialFileNotFound` when offline).
- **Orekit data** is resolved before the JVM starts (`orekit_utils.resolve_orekit_data()`),
  kept in the cache and reused offline (`OrekitDataNotFoundError` when there is none).
- `contigo.plotting` package skeleton (decimation, antimeridian splitting, per-spacecraft
  colours); matplotlib is an optional extra (`pip install "contigo[plot]"`).
- `contigo.__version__`.
- Test suite (about 370 fast tests, plus `network`, `orekit` and `slow` tiers) and GitHub
  Actions CI (ruff, black, pytest on Python 3.10, 3.12, 3.13 and 3.14).
- Documentation: installation and data, coordinates, writing a force model, testing,
  contributing.

### Changed

- **Downloads go to the per-user cache, never into the installed package.** This covers SPICE
  kernels, gravity models and Orekit data.
- The Orekit data zip is **no longer tracked in git**. A fresh clone downloads it (~21 MB) on
  first Orekit use. Pulling this change deletes the tracked copy from existing checkouts; that
  is expected.
- The Orekit helper jar moved to `contigo/java/` and ships as package data, so non-editable
  installs work.
- Importing contigo never starts the JVM or uses the network (`srp_orekit` and `jb08_utils`
  used to at import).
- contigo no longer configures logging (`logging.basicConfig` removed) and uses `logging`
  instead of `print`. Call `logging.basicConfig(level=logging.INFO)` to see progress messages.
- Downloads have timeouts and are atomic: an interrupted download never leaves a partial file
  in place.
- `time_utils.spice_time` accepts time scales in any case and supports UTC to UTC.
- Packaging: all subpackages are installed (`packages.find`); `tables` (PyTables) replaces the
  unused `h5py`; `platformdirs` added; the `git+https` `orekit` extra removed (Orekit is a core
  dependency).
- Tutorials moved to `notebooks/tutorials/`, analysis notebooks to `notebooks/analysis/`, old
  development notebooks to `notebooks/archive/`.

### Deprecated

- `ThirdBodyAcc`: use `ThirdBodyEnv` with a `SolarSystemEnvironment`. Removed in 0.3.0.
- The Orekit helper jar in the old location `java_src/target/` (`FutureWarning`), and an Orekit
  data zip in `contigo/data/` (used only when the cache has none). Removed in 0.3.0.

### Removed

- `ForceModel.is_conservative` (nothing used it).

### Fixed

- `Spacecraft(..., unit_input='km')` no longer multiplies the caller's array in place.
- Reloading a `Spacecraft` recomputes its time axes.
- SP3 loading (paths, multi-satellite epochs); duplicate (ID, time) rows from overlapping files
  are dropped when the IDs are real.
- `GravPot`: no out-of-bounds read when `lmax` exceeds the file (clamped); clear errors.
- `SolarSystemEnvironment` asks the provider once per missing time bin.
- `EDRDensity`: force models with the same `name` no longer overwrite each other.
- EFD windows now use the full window (a 900 s window spanned 890 s).
- `ThirdBodyAcc` works with a pandas `Series` time.
- An interrupted SPICE kernel download no longer leaves a truncated kernel in the cache (which
  broke every later run); a failed refresh falls back to the kernel already on disk.
- The `IAU_EARTH` fallback warning points at the caller's line; `ecef_to_inertial` accepts
  empty input.
- `CONTIGO_Colab` tutorial updated to the current API.

## [0.1.0]

First version: `Spacecraft`, `Constellation`, SPICE and Orekit ephemeris providers,
`SolarSystemEnvironment`, third-body, SRP and gravity-potential force models, and
`EDRDensity` (EDR, denominator and effective density).
