# To do

Open work that is not part of a current plan. Move an item into a plan (or a PR) when someone
starts it, and delete it here when it is done.

## Soon

### Remove GMAT (was R15)

GMAT is hard to install and work with, and Orekit covers the same ground (`SRPOrekitCB`).
Scope:

- `contigo/forces/srp_gmat.py` (`SRPAcc`, `SRPGMATAcc`) and `contigo/forces/srp_utils.py`;
- `config.state['gmat_loaded']` and `config.state['gmatpy']`;
- GMAT mentions in `README.md`, `CLAUDE.md`, `contigo/forces/__init__.py` and
  `docs/force_models.md`;
- `notebooks/tutorials/CONTIGO_Colab.ipynb`: its GMAT install cells. Switch it to
  `SRPOrekitCB`, or fold it into `Tutorial_Overview`;
- `data/efd_gmat_ekf.hdf` (used only by `notebooks/archive/testing_forces.ipynb`).

Path: a `DeprecationWarning` on `SRPAcc` in the next release, removal in the one after (as for
`ThirdBodyAcc`). Or remove it directly, since no tests or tutorials other than `CONTIGO_Colab`
depend on it. Decide before starting.

### Plotting task 2.1: package the coastline file

`plot_ground_track` will read a bundled coastline (`contigo/plotting/data/coastline.npz`). To
ship it in the installed package:

- add `"plotting/data/*.npz"` to `[tool.setuptools.package-data]` in `pyproject.toml`;
- add the file to the CI install check in `.github/workflows/ci.yml` (the step that asserts
  the bundled files exist), so a missing file fails CI;
- record its source and licence (Natural Earth, public domain) next to the file or in the
  module docstring; keep it under ~100 KB.

### The Orekit test tier needs the network on a fresh machine

`pytest --run-orekit` fails with an empty cache (for example `EP-06b` raises
`OrekitDataNotFoundError`). Since the Orekit data zip is no longer tracked, the tests download
it into their temporary cache, but the fast-tier guard blocks the network for any test not
marked `network`. Options: let `orekit`-marked tests use the network in `tests/conftest.py`,
or copy the user's cached zip into the test cache when one exists.

## Before the 0.2.0 release

- **Refresh the bundled Earth orientation kernel** (`contigo/data/earth_latest_high_prec.bpc`,
  5 MB) from NAIF, so the bundled coverage reaches past 2026-05-30. Update the end date in
  `docs/coordinates.md`, `docs/installation.md`, `docs/plotting_plan.md` and the tests that
  use it (`tests/test_earth_orientation.py` EO-01 and EO-04, `ET_2026_09` in
  `tests/test_coords.py`).
- Re-run `notebooks/tutorials/CONTIGO_Colab.ipynb` in Colab (it needs GMAT and cannot run in
  CI).
- Tag the release (`v0.2.0`) and date the CHANGELOG entry.

## For 0.3.0

Remove what 0.2.0 deprecated:

- `ThirdBodyAcc` (`forces/third_body_acc.py`) and its test FO-12;
- the old jar location `java_src/target/` in `orekit_utils.helper_jar_path()`;
- the old Orekit data zip location `contigo/data/` in `orekit_utils.resolve_orekit_data()`.

## Later

- **One atomic-download helper.** The pattern "download to a per-process `.part` file, check
  it, `os.replace`, clean up" exists three times: `kernels._download`,
  `orekit_utils.resolve_orekit_data` and `grav_utils.fetch_icgem`, with small differences (how
  cleanup errors are handled). Move it into one `utils.dl_file_atomic(url, target,
  validate=None)` and use it in all three.
- **Deduplicate in `load_from_arrays` and `split_by_id`.** SP3 loading drops duplicate (ID,
  time) rows, but arrays and CSV/HDF files with duplicates are kept as given, which can break
  the Simpson integration (zero time steps).
- **Tighten ED-08** (full ESA orbit vs Orekit EDR, `atol=1e-7, rtol=0.1` today). The archived
  `testing_forces` notebook matched Orekit to ~15 significant figures with GMAT SRP.
- **Untested modules:** `hasdm_utils` (needs a sample HASDM file) and `jb08_utils` (Orekit
  tier; only input validation is tested).
- **Run `mypy` in CI.** It is in the `dev` extra and CLAUDE.md, but not checked.
- **Data lookup order:** `config.find_data_file` looks in the current folder first, so a stray
  `naif0012.tls` or `.gfc` file in the working directory wins over the cache and the bundled
  copy. Intended, but worth documenting in error messages or reconsidering.
- **Docs site** (`mkdocs`, extra `docs`) built from `docs/`.
- **Publish to PyPI** once the GMAT decision is made.
