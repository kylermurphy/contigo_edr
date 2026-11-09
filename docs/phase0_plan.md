# CONTIGO Phase 0: Foundations, Testing and Clean-up Plan

| | |
|---|---|
| **Status** | Draft v2: Kyle's answers to the v1 open questions applied (2026-11-09) |
| **Parent plan** | [`plotting_plan.md`](plotting_plan.md), Phase 0 |
| **Owners** | Kyle Murphy (0.2, 0.3, 0.6), Claude Code (0.1, 0.4, 0.5a, 0.5b, refactors) |
| **Reviewer** | Kyle Murphy (every PR) |

> Markers: **[DECISION]** = agreed; **[PROPOSED]** = Claude's recommendation, needs Kyle's OK;
> **[OPEN]** = needs input. Edit this file in a PR or comment on the PR that adds it.

---

## 0. Decisions already made (2026-11-09)

| # | Decision |
|---|----------|
| P1 | **[DECISION]** Large gravity-field files (`EGM2008.gfc`, `EIGEN-6C4.gfc`) are **downloaded on demand** from ICGEM into `contigo/data/`; only `EIGEN-2.gfc` stays bundled. |
| P2 | **[DECISION]** Tests use **pytest markers** (`network`, `orekit`, `slow`). CI runs only the fast, unmarked tier: pure NumPy/SPICE with bundled kernels, synthetic fixtures and small trimmed reference data. The marked tiers run locally on request. |
| P3 | **[DECISION]** The `testing_*` notebooks are **mined for checks, then archived** in `notebooks/dev/` with outputs cleared. A notebook is deleted only if the new tests fully cover it. |
| P4 | **[DECISION]** **Keep all three tutorials** (`Tutorial_Overview`, `CONTIGO_Colab`, `Orekit_EDR`), moved into `notebooks/tutorials/`. |
| P5 | **[DECISION]** Rename the duplicate "0.5": CI + tests stay **0.5**; the SPIRE dataset (Kyle) becomes **0.6**. |
| P6 | **[DECISION]** Claude takes task 0.1 (packaging) as the first PR. |
| P7 | **[DECISION]** **Sequencing:** Kyle starts 0.2/0.3/0.6 only after all of Claude's Phase 0 work is merged. Claude's PRs therefore have no file conflicts with Kyle's and can touch any module. |
| P8 | **[DECISION]** **PR workflow:** Claude prepares each branch and shows Kyle what changed (diff summary + key hunks) plus suggested PR title and body text. **Kyle pushes and opens every PR himself.** Claude and its subagents never push or open PRs. |
| P9 | **[DECISION]** Orekit stays a **core** dependency (not an extra). |
| P10 | **[DECISION]** GMAT is **likely to be removed** (hard to work with). No GMAT tests are written in Phase 0; the GMAT ones are listed as ToDo (§4.8) and the removal is refactor R15. |
| P11 | **[DECISION]** CI matrix: Python 3.10, 3.12, 3.13, 3.14. |
| P12 | **[DECISION]** Each subagent runs on an explicitly chosen model matched to the task (§6.1). |

---

## 1. Who does what

| Task | Owner | Depends on | Summary |
|------|-------|-----------|---------|
| **R1** | Kyle, **now** | none | Unstage `data/EGM2008.gfc` (254 MB) and `data/EIGEN-6C4.gfc` (180 MB). GitHub rejects files over 100 MB, so the next push would fail. `git restore --staged data/*.gfc`, then add them to `.gitignore`. |
| 0.1 | Claude | R1 | Packaging: find subpackages, add the `plot` and `test` extras, ship the jar and data as package data, add `tables`, create a `tests/` skeleton |
| 0.2 | Kyle | all Claude PRs merged (P7) | `ecef_to_geodetic` and WGS-84 constants (as in the parent plan) |
| 0.3 | Kyle | all Claude PRs merged (P7) | `check_pck()` and `ecef_to_inertial` (as in the parent plan) |
| 0.5a | Claude | 0.1 | GitHub Actions workflow: ruff, black --check and fast-tier pytest on 3.10, 3.12, 3.13, 3.14 |
| 0.5b | Claude | 0.5a | Test suite for the existing `contigo` code (Section 4), written in stages |
| 0.4 | Claude | 0.1 | `contigo/plotting/` skeleton: `_style.py`, `_util.py`, lazy matplotlib import, tests |
| 0.6 | Kyle | all Claude PRs merged (P7) | New SPIRE ECEF/GPS dataset, loadable into `Spacecraft`/`Constellation` |
| R2 to R13 | Claude (Kyle approves each) | the tests that cover them | Refactors and deprecations (Section 5) |
| R15 | Kyle decides; Claude implements if approved | Phase 0 tests merged | Remove GMAT |

Suggested order of Claude's PRs (each one under ~300 changed lines where possible):

```
R1 (Kyle) ─► 0.1 packaging ─► 0.5a CI ─┬─► T0 fixtures/conftest ─┬─► T1..T6 test PRs (parallel)
                                       │                         └─► refactor PRs (each after its tests)
                                       └─► 0.4 plotting skeleton (parallel with T-PRs)
                                 ...all of the above merged ─► Kyle starts 0.2, 0.3, 0.6
```

Every PR follows P8: Claude shows the changes and suggested PR text; Kyle pushes and opens it.

---

## 2. Findings from the code review

These affect Phase 0 directly. Each one links to a test or refactor ID.

| ID | Finding | Where | Action |
|----|---------|-------|--------|
| F1 | Two staged `.gfc` files exceed GitHub's 100 MB limit | `data/` | R1, R2 |
| F2 | `packages = ["contigo"]` drops subpackages from a non-editable install. `package-data` names `contigo.orekit_utils`, which does not exist. `start_orekit()` finds the jar through `../../../java_src/target`, which only works in an editable checkout | `pyproject.toml`, `orekit_utils.py` | 0.1, R5 |
| F3 | `pandas.read_hdf` needs **PyTables** (`tables`), which is not declared. `h5py` is declared but never imported | `pyproject.toml` | 0.1 |
| F4 | `forces/srp_orekit.py` **starts the JVM and contacts GitLab at import time**. Any test collection or `import` that reaches it needs Java and the network | `srp_orekit.py` | R3, PK-01 |
| F5 | `SPICEEphem.load_kernels()` sends an HTTP HEAD request to NAIF on first use, even when every kernel is already bundled. `start_orekit()` always queries the GitLab API | `spice_ephem.py`, `orekit_utils.py` | R11 (offline switch), test fixture |
| F6 | `ForceModel` Protocol declares `is_conservative: bool`, but no model defines it. Because the Protocol is `@runtime_checkable`, `isinstance(model, ForceModel)` returns **False** for every model | `forces/base.py` | FO-09, R6 |
| F7 | `GravPot.get_pot()` returns `None` and does not raise when `calc_pot()` was never called (`gravpot` is set to `None` in `__init__`, so the `hasattr` check always passes). When coefficients come from the `config.state` cache, `lmax` is not clamped to the file maximum | `grav_pot.py` | FO-07, R7 |
| F8 | SP3 loading is broken: `df_sp3` calls `fn.lower()` on a `Path`, accepts no `**kwargs`, and has a placeholder for multi-satellite epochs | `utils.py`, `spacecraft.py` | SC-11, R8 |
| F9 | `CONTIGO_Colab.ipynb` no longer runs. It imports `ThirdBody` (which does not exist) and passes `sp_et=`/`sp_gps=` to `SolarSystemEnvironment` (now `ephem_time=`/`gps_time=`/`utc_time=`) | notebook | NB-01, N2 |
| F10 | `config.py` calls `logging.basicConfig` at import, which changes the user's root logger. Several modules use `print` | `config.py`, `orekit_utils.py`, `grav_pot.py` | R4 |
| F11 | `ThirdBodyAcc` (legacy) duplicates `ThirdBodyEnv` + `SolarSystemEnvironment` and has its own time-conversion code | `third_body_acc.py` | R9 (deprecate) |
| F12 | `contigo/data/orekit_data_<sha>.zip` (21 MB) is tracked and is renamed on every upstream orekit-data commit, which bloats the repository | `contigo/data/` | R5 |
| F13 | ruff config uses the deprecated top-level `select`. The existing code has never been linted, so a strict `ruff check contigo` will start red | `pyproject.toml` | 0.5a, R13 |
| F14 | `forces/__init__.py` has the third-body module docstring pasted in | `forces/__init__.py` | R12 |
| F15 | `time_utils.spice_time` does not upper-case `out_scale` and loops over `unitim` element by element | `time_utils.py` | TU-03, R10 |

---

## 3. Phase 0 tasks owned by Claude

### 0.1 Packaging **[DECISION owner: Claude]**

- `[tool.setuptools.packages.find] include = ["contigo*"]`.
- `package-data`: `contigo = ["data/*.tls", "data/*.bpc", "data/*.bsp", "data/EIGEN-2.gfc", "java/*.jar", "py.typed"]`. Move the jar to `contigo/java/` (R5 makes `start_orekit` find it through `importlib.resources`; until then keep the old path as a fallback).
- Extras: `plot = ["matplotlib>=3.7"]`; `test = ["pytest", "pytest-cov"]`, folded into `dev`.
- Dependencies: add `tables`; drop `h5py` unless Kyle wants it kept.
- ruff: move the settings to `[tool.ruff.lint]`.
- pytest: register the markers and add `tests/conftest.py` with `--run-network`, `--run-orekit`, `--run-slow` and `--run-all` options (marked tests are skipped unless their flag is given). No `gmat` marker (P10).
- `tests/test_smoke.py`: `import contigo`, `contigo.config.DATA_DIR` exists.

**Acceptance:** in a fresh venv, `pip install .` (non-editable) gives working `import contigo.forces`, `contigo/data/naif0012.tls` is present in site-packages, and `pytest` passes.

### 0.5a CI workflow

`.github/workflows/ci.yml`, run on `pull_request` and `push` to `main`:

| Job | Python | Steps |
|-----|--------|-------|
| `lint` | 3.12 | `ruff check contigo tests`, `black --check contigo tests` |
| `test` | 3.10, 3.12, 3.13, 3.14 | `pip install ".[dev,plot]"`, `pytest --cov=contigo` (fast tier only; `CONTIGO_OFFLINE=1`) |
| `install` | 3.12 | non-editable `pip install .` + `python -c "import contigo.forces, contigo.ephemeris"` (PK-02) |

- **ruff on existing code [PROPOSED]:** CI lints `tests/` and `contigo/plotting/` strictly from day one. The rest of the package goes into a per-file `extend-exclude` list, which R13 (a mechanical `ruff --fix` PR) empties.
- Orekit is a core dependency (P9), so CI installs the `jdk4py` JDK wheel but the fast tier never starts the JVM.
- 3.13/3.14: if a dependency (numba, pyshtools, jdk4py) has no wheel yet for a version, that matrix entry is temporarily marked `continue-on-error` and noted in the PR. It is not silently dropped.
- Cache pip, and set `timeout-minutes: 15`.

**Acceptance:** a PR shows green or red checks; branch protection (plotting plan §5.4) requires `lint` and `test`.

### 0.5b Test suite: see Section 4.

### 0.4 Plotting skeleton

Files: `contigo/plotting/__init__.py`, `_style.py`, `_util.py`, `tests/plotting/test_util.py`, `test_style.py`.

| Function | Behaviour | Tests |
|----------|-----------|-------|
| `_util.as_spacecraft_dict(data, ids=None)` | `Spacecraft` → `split_by_id()`; `Constellation` → `.spacecraft`; `ids` filters with string-safe matching (IDs are pandas categories); unknown ID → `ValueError` naming the available IDs; any other type → `TypeError` | PL-01..04 |
| `_util.decimate(n, max_points)` | `n <= max_points` → `arange(n)`; otherwise unique evenly spaced indices that include `0` and `n-1`; `max_points < 2` → `ValueError` | PL-05..07 |
| `_util.split_antimeridian(lon, *arrays)` | Insert `NaN` where `abs(diff(lon)) > pi` in lon and in every companion array | PL-08 |
| `_style.color_for(sc_id)` | Deterministic colour across runs and processes (`zlib.crc32(str(id))` into `tab20`), **not** `hash()`, which is randomised per process | PL-09 |
| `_style.rc_context()` | Shared style through `plt.rc_context` with no global state change | PL-10 |
| `_style.require_matplotlib()` | Lazy import; when matplotlib is missing raises `ImportError("... pip install 'contigo[plot]'")` | PL-11 |

**Acceptance:** `import contigo.plotting` works without matplotlib (test simulates this with `monkeypatch.setitem(sys.modules, "matplotlib", None)`), and all PL tests pass.

---

## 4. Test catalogue (0.5b), for review

**How to review:** for each row, mark it keep / change / drop, or edit the expected value or
tolerance. Rows marked **xfail** pin a known bug with `pytest.mark.xfail(strict=True)`, so the
test turns green, and CI forces removal of the marker, when the matching refactor lands.
"Source" names the notebook the check was mined from (to be completed by the mining step in §6.2).

### 4.0 Shared fixtures (`tests/conftest.py`, PR **T0**)

| Fixture | Contents |
|---------|----------|
| `spice_offline` (session, autouse) | `furnsh` bundled `naif0012.tls`, `earth_latest_high_prec.bpc`, `de440s.bsp`; set `config.state['kernel_downloaded']=True`; `monkeypatch` `utils.dl_file`/`wf_mtime` to fail loudly, so no unmarked test can reach the network |
| `restore_config_state` (autouse) | Snapshot and restore `config.state` (the potential-coefficient cache otherwise leaks between tests) |
| `circular_orbit(alt, inc, n, dt, t0)` | Factory: inertial circular Keplerian orbit, rotated to ECEF with `constants.WGS84_EARTH_ANGULAR_VELOCITY` (`v_ecef = R(v_i − ω×r_i)`). Analytic, exact |
| `sc_single`, `sc_multi` | `Spacecraft` from arrays; `sc_multi` has 3 IDs with **shuffled** times and per-ID properties |
| `constellation_fixture` | `Constellation(sc_multi)` |
| `FakeProvider` / `FakePotential` / `FakeForce` | Call-counting ephemeris provider; point-mass potential `GM/r`; force `a = −k·v` |
| `esa_pod_small`, `orekit_edr_small` | Trimmed ~2 orbits (~1,100 rows at 10 s) of `data/ESA_pod.hdf` and the matching rows of `data/ore_d.hdf`, saved as `tests/data/*.csv.gz` (< 500 KB total), plus `tests/data/make_fixtures.py` that regenerates them |

### 4.1 Spacecraft (`tests/test_spacecraft.py`, PR **T1**)

| ID | Test | Expected |
|----|------|----------|
| SC-01 | Load from arrays | `N`, shapes, default `sc_id == "NO_ID"` |
| SC-02 | `unit_input='km'` | positions and velocities × 1000 |
| SC-03 | Bad input | `ValueError` for: no `tscale_input`, bad scale, state `(N,5)`, length mismatch, state without time, neither arrays nor file |
| SC-04 | Property broadcasting | scalar → `(N,)`; length-N array kept; wrong length → `ValueError`; missing → NaN |
| SC-05 | CSV and HDF loaders (written to `tmp_path`) | custom column names, glob over 2 files, `sc_id_col='filename'` + `sc_fn_slc` |
| SC-06 | Loader errors | missing column → `ValueError`; no files → `FileNotFoundError`; unknown extension → `ValueError` |
| SC-07 | `split_by_id` | one object per ID, each sorted by GPS time, properties sliced consistently, total N preserved |
| SC-08 | Time axes | GPS input: `|sspice_et − sspice_gps − 51.184| < 2e-3` s |
| SC-09 | Leap seconds | UTC 2017-01-01T00:00:00 equals GPS 2017-01-01T00:00:18 (and +13 s in 2005, the CHAMP era) |
| SC-10 | `spherical()` | known points on the x-axis, y-axis and pole → `[r, lat, lon]` |
| SC-11 | SP3 loader (synthetic 1-sat and 2-sat SP3 in `tmp_path`) | km → m, dm/s → m/s, correct epochs. **xfail** until R8 |
| SC-12 | `state_data` cache | invalidated after `load_from_arrays` |

### 4.2 Constellation (`tests/test_constellation.py`, PR **T1**)

| ID | Test | Expected |
|----|------|----------|
| CO-01 | From kwargs and from a multi-ID `Spacecraft` | `len`, `ids`, union time axes sorted and unique |
| CO-02 | From a list of `Spacecraft` | merge works; duplicate IDs → `ValueError`; empty list → `ValueError`; non-Spacecraft → `TypeError` |
| CO-03 | `+` and `+=` | the three union time arrays stay aligned (`et − gps ≈ 51.184` everywhere); `+` does not modify its operands |
| CO-04 | Container protocol | `__iter__`, `__getitem__`, `__len__`, `repr` |

### 4.3 Time and utilities (`tests/test_time_utils.py`, `tests/test_utils.py`, PR **T2**)

| ID | Test | Expected |
|----|------|----------|
| TU-01 | GPS J2000 epoch → ET | `51.184 ± 2e-3` s |
| TU-02 | UTC → GPS across leap seconds | offset 13 s (2005), 17 s (2016), 18 s (2017) |
| TU-03 | Scale validation | bad scale → `ValueError`; lower-case `out_scale` is accepted (**[DECISION]**; **xfail** until R10) |
| TU-04 | `out_scale='UTC'` | `ValueError` (not implemented) |
| TU-05 | `check_lpsk` | idempotent (`ktotal` unchanged on a second call); download path tested with a monkeypatched `dl_file` that copies the bundled file |
| UT-01 | `wf_mtime` | parses `Last-Modified`; returns `None` on error (mocked `requests.head`) |
| UT-02 | `df_sp3` | see SC-11 |

### 4.4 Ephemeris and environment (`tests/test_ephemeris.py`, PR **T3**)

| ID | Test | Expected |
|----|------|----------|
| EP-01 | `SPICEEphem` Sun/Moon | shape `(B,N,3)` in metres; Sun 1.471e11 to 1.521e11 m; Moon 3.56e8 to 4.07e8 m |
| EP-02 | Duplicate epochs | identical rows (unique/inverse logic) |
| EP-03 | `load_kernels` twice | no duplicate furnsh |
| EP-04 | `SolarSystemEnvironment` cache (FakeProvider) | repeated times → 0 extra provider calls; only missing bins requested; tolerance binning; `tolerance=None` and `0.0`; `N=1` case |
| EP-05 | GM lookup | from `provider.ephemeris`; unknown body raises (proposed: clear `ValueError`) |
| EP-06 | `orekit` | `OrekitEphem` and `SPICEEphem` Sun/Moon agree to within ~1 km (**[DECISION]**) |

### 4.5 Forces (`tests/forces/`, PRs **T4**, **T5**)

| ID | Test | Expected |
|----|------|----------|
| FO-01 | `tba_pairwise_numba` | equals a NumPy reference on random inputs (rtol 1e-12); analytic collinear case `μ(1/(D−r)² − 1/D²)` |
| FO-02 | `ThirdBodyEnv` + FakeProvider | dict keyed by constellation IDs, `(B,N,3)` |
| FO-03 | Sun+Moon at LEO (real SPICE) | tidal magnitude in 1e-7 to 2e-6 m/s² |
| FO-04 | `read_icgem_coeff(EIGEN-2)` | `lmax=140`, `GM=3.986004415e14`, `r0=6378136.46`, `C00=1`, `C̄20≈−4.8417e-4`; a file with no header → `ValueError` |
| FO-05 | `get_potential_batch` | `lmax=0` → `GM/r`; `lmax=2` (zonal only) → `GM/r·[1 + (r0/r)² C̄20 √5 (3sin²φ−1)/2]` (rtol 1e-12) |
| FO-06 | Scalar vs batch | `get_potential` equals `get_potential_batch` |
| FO-07 | `GravPot` | `get_pot` before `calc_pot` raises (**xfail**, R7); `lmax` above the file maximum is clamped on the cached path too (**xfail**, R7); missing file → `ValueError` |
| FO-08 | `EarthPotential` | dict of `(N,)` arrays; `acceleration` → `NotImplementedError` |
| FO-09 | Protocol conformance | `isinstance(m, ForceModel)` for every model (**xfail**, R6) |
| FO-10 | ~~GMAT SRP~~ | **ToDo, not written** (P10, §4.8) |
| FO-11 | `orekit` | `SRPOrekitCB` shape `(N,3)`; roughly anti-sunward; magnitude consistent with `Cr·A/m` (~4.56e-6 N/m² at 1 AU) |
| FO-12 | `ThirdBodyAcc` | equals `ThirdBodyEnv` result; after R9, emits `DeprecationWarning` |

### 4.6 EDR / EFD (`tests/test_edr_efd.py`, PR **T6**)

The **physics tests** use no data. For a Keplerian orbit under a point-mass potential, the
Jacobi integral `v²/2 − U − ω²(x²+y²)/2` is conserved in the rotating (ECEF) frame, so the
EDR must stay at zero.

| ID | Test | Expected |
|----|------|----------|
| ED-01 | `circular_orbit` (inclined) + FakePotential, no forces | `max|EDR| < 1e-6 m²/s²` (Jacobi integral conserved) |
| ED-02 | Equatorial orbit + `FakeForce(a=−k·v)` | `EDR(t) = k·|v_ecef|²·t` (checks the sign convention and the Simpson integration) |
| ED-03 | One `(B,N,3)` force vs B separate `(N,3)` forces | identical EDR |
| ED-04 | `compute_denom`, constant `b` and `|v|` | `b·|v|³·t` |
| ED-05 | `compute_den` | window count `floor(T/window)`; midpoint `DateTime`; `efd = −2k/(b|v|)` for ED-02/ED-04 inputs; mismatched keys → `ValueError` |
| ED-06 | Constructor | warms the env cache with exactly one provider call |
| ED-07 | Regression snapshot (`esa_pod_small`, ThirdBodyEnv+SPICE, EarthPotential EIGEN-2 lmax 50) | matches `tests/data/edr_snapshot.npz` (rtol 1e-10). Mark `slow` if it takes over 10 s |
| ED-08 | `orekit` validation: full `ESA_pod` vs `ore_d`, the CONTIGO_Colab check but with `SRPOrekitCB` in place of GMAT `SRPAcc` | `allclose(atol=1e-7, rtol=0.1)`. **[DECISION]** keep it loose for now; tighten later |

> ED-07 is a *regression* test (did the output change?), not a *validation* test (is it right?).
> ED-01/02/04/05 and ED-08 provide the validation.

### 4.7 Package and notebooks (`tests/test_package.py`, PR **T2**)

| ID | Test | Expected |
|----|------|----------|
| PK-01 | Import every submodule (`pkgutil.walk_packages`) | no import starts the JVM (`jpype.isJVMStarted()` stays False). `srp_orekit` is **xfail** until R3 |
| PK-02 | CI `install` job | non-editable install imports and finds data and the jar |
| NB-01 | Every `from contigo... import X` in `notebooks/tutorials/*.ipynb` resolves (static parse, no execution) | catches F9-type breakage. `CONTIGO_Colab` is **xfail** until N2 |

### 4.8 Out of scope for Phase 0 (later)

`hasdm_utils` (needs a sample HASDM file), `jb08_utils` (Orekit tier), executing tutorials
end-to-end (`nbmake`, local `--run-all` only), performance benchmarks.

**GMAT ToDo (P10), not written now; delete this list if R15 removes GMAT:**
FO-10 (`SRPAcc`/`SRPGMATAcc` direction and magnitude), GMAT vs Orekit SRP agreement,
`srp_utils.setup_gmat` path handling, the GMAT route of ED-08. PK-01 still imports `srp_gmat`
(the import does not load GMAT), so the module keeps working until a decision is made.

---

## 5. Refactors, deprecations and notebook clean-up

Each refactor lands **after** the tests that cover it, in its own PR, and removes the matching xfail.

| ID | Change | Priority | Unblocks |
|----|--------|----------|----------|
| R1 | Unstage the big `.gfc` files (Kyle) | **Now** | pushing |
| R2 | ICGEM download on demand: a small registry `{name: url}` in `grav_utils`; `GravPot` downloads into `DATA_DIR` when the file is missing; delete the duplicate `data/EIGEN-2.gfc` | High | P1 |
| R3 | `srp_orekit.py`: move the JVM start and Java imports into `__init__` (same pattern as `OrekitEphem`) | High | PK-01, CI |
| R4 | `config.py`: drop `logging.basicConfig`; replace `print` with `logger` | Medium | |
| R5 | `orekit_utils`: jar via `importlib.resources`; if GitLab is unreachable, use the newest local `orekit_data_*.zip`; raise instead of returning an error string on 404; untrack `contigo/data/orekit_data_*.zip` (add to `.gitignore`) | High | non-editable install, offline use |
| R6 | `ForceModel.is_conservative`: **[PROPOSED]** remove it from the Protocol (nothing reads it) | Medium | FO-09 |
| R7 | `GravPot`: raise in `get_pot` when nothing is computed; clamp `lmax` on the cached path; key the cache on the full path | Medium | FO-07 |
| R8 | `df_sp3`: accept `Path`, `**kwargs`, multi-satellite epochs (track the current epoch inside the parse loop) | Medium, **fix now** (**[DECISION]**) | SC-11 |
| R9 | **Deprecate `ThirdBodyAcc`**: `DeprecationWarning` in 0.2.0 pointing to `ThirdBodyEnv`; remove in 0.3.0. Keep `GravPot` (the engine behind `EarthPotential`) and scalar `get_potential` (useful as the reference in FO-06). `SRPGMATAcc` is handled by R15 | Medium | FO-12 |
| R10 | `time_utils`: upper-case `out_scale`; numpy docstrings; vectorisation optional | Low | TU-03 |
| R11 | `CONTIGO_OFFLINE=1` env switch read in `config`: skips the NAIF mtime check and the GitLab SHA query, using local files | High | CI determinism, Colab cold start |
| R12 | Fix the `forces/__init__.py` docstring; add `contigo.__version__` | Low | |
| R13 | Mechanical `ruff --fix` + `black` over `contigo/`, then empty the ruff exclude list | Medium (separate PR, no logic changes) | strict lint |
| ~~R14~~ | ~~Move Orekit to an extra~~: **dropped**, Orekit stays core (P9) | | |
| R15 | **Remove GMAT** (P10, Kyle's call). Scope: `forces/srp_gmat.py` (`SRPAcc`, `SRPGMATAcc`), `forces/srp_utils.py`, `config.state['gmat_loaded'/'gmatpy']`, GMAT references in README/CLAUDE.md, and the GMAT install cells in `CONTIGO_Colab` (which would switch to `SRPOrekitCB` or be merged into `Tutorial_Overview`). Path: `DeprecationWarning` on `SRPAcc` in 0.2.0, removal in 0.3.0, same as R9 | **ToDo**, after Phase 0 | simpler install, fewer tutorials |

### 5.1 Notebook layout **[PROPOSED]**

```
notebooks/
  tutorials/        # maintained, Colab badges, small outputs only
    Tutorial_Overview.ipynb        (main entry point)
    CONTIGO_Colab.ipynb            (GMAT + SPICE route; fix F9)
    Orekit_EDR.ipynb               (Orekit route)
    Plotting_Examples.ipynb        (Phase 1 of plotting plan)
  analysis/         # science notebooks, may need external data
    champ_stats.ipynb
  dev/              # archived; outputs cleared; header cell says "archived, see tests/..."
    ...testing_* that are not fully covered
```

| Notebook | Size | Proposed fate (confirmed after mining, §6.2) |
|----------|------|----------------------------------------------|
| `testing_sc` | 19 KB | mine → SC/CO tests → **delete** if fully covered |
| `testing_force_model_constellation` | 17 KB | mine → ED/FO → **delete** if covered |
| `testing_ephem` | 51 KB | mine → EP/FO-03/FO-12 → **delete** if covered |
| `testing_orekit_ephem` | 13 KB | mine → EP-06 → **delete** if covered |
| `testing_forces` | 216 KB | mine → FO; GMAT/Orekit comparisons → **archive** |
| `test_edr_efd` | 95 KB | mine → ED; **archive** |
| `testing_swarm`, `testing_champ` | 644 / 442 KB | science validation with `D:\` data → **archive** |
| `champ_stats` | 197 KB | → `analysis/` |

- N1: move the tutorials; update Colab badge URLs (`.../blob/main/notebooks/tutorials/...`) and README links.
- N2: fix `CONTIGO_Colab` imports and `SolarSystemEnvironment` kwargs only (minimal, since R15 may rewrite or retire it). Kyle re-runs it in Colab, since it needs GMAT.
- N3: **[PROPOSED]** add `nbstripout` as a pre-commit hook, or at least a CI check that notebooks in `tutorials/` are under 1 MB.

---

## 6. Agents: who runs what, and how to dispatch

The **main Claude Code session** (Kyle + Claude, in VS Code, Opus) is the orchestrator. It owns
this plan and the local branches. Subagents do bounded, well-specified work and report back.
Subagents commit only locally, inside their own worktree branch, so each PR has a clean diff.
**Nobody but Kyle pushes or opens a PR** (P8).

### 6.1 Agent roster and models (P12)

The `model` parameter is set **explicitly on every dispatch**. Rule of thumb: Haiku for mechanical
edits with no judgment, Sonnet for well-specified code and reading, Opus for physics, numerics,
design and anything where a subtle wrong answer would pass review.

| Agent type | Model | Used for | Isolation | Parallelism |
|------------|-------|----------|-----------|-------------|
| `Explore` (read-only, "medium") | `sonnet` | Mining the old notebooks for checks (§6.2): it has to read the science and map each check to a catalogue ID | none (read-only) | up to 3 at once, background |
| `general-purpose` | `sonnet` | Test PRs with straightforward expected values: **T1** (Spacecraft/Constellation), **T2** (time/utils/package), **T3** (ephemeris/cache); the plotting skeleton **0.4**; small refactors **R4, R6, R10, R12** | `isolation: "worktree"` | up to 3 at once; never two on the same files |
| `general-purpose` | `opus` | Physics and numerics: **T4/T5** (forces: potential closed form, third body), **T6** (EDR/EFD analytic tests, regression snapshot); refactors with design risk **R2, R3+R5, R7, R8, R9, R11** | `isolation: "worktree"` | same pool as above |
| `general-purpose` | `haiku` | Mechanical work: notebook moves, clearing outputs, badge/README link updates (**N1, N3**); applying `ruff --fix`/`black` (**R13**) | `isolation: "worktree"` | 1 |
| `Plan` | `opus` | Design review before R2, R3+R5, R11 (and R15 if approved) | none | one at a time |
| `/code-review` skill (`medium`) | main session | Run on each finished branch before it goes to Kyle | n/a | n/a |
| `fork` | (n/a) | Not used; worktree agents give clean diffs per PR | | |

The main session (not a subagent) writes **T0** (shared fixtures), because every test depends on
it and the analytic orbit fixture must be right. Do not use `max`/`ultra` review levels unless
Kyle asks. Reasoning effort is left at each agent's default unless Kyle asks otherwise.

### 6.2 Step A: mine the notebooks (Explore, parallel, background)

Dispatch three `Explore` agents (`model: "sonnet"`, `run_in_background: true`), one per group:
(a) `testing_sc`, `testing_force_model_constellation`;
(b) `testing_forces`, `testing_ephem`, `testing_orekit_ephem`;
(c) `test_edr_efd`, `testing_swarm`, `testing_champ`, `champ_stats`.

Prompt template:

```
Read-only task. Read the notebooks <list> in D:\GitHub\contigo_edr (use
`python -c` with json to print code cells and short text outputs; skip images).
For every check the author performed (printed comparisons, np.allclose, asserts,
plotted comparisons with a stated expectation), report one row:
| notebook:cell | what is checked | inputs/data used | expected value/tolerance |
| already covered by catalogue ID in docs/phase0_plan.md §4? (ID or "NEW") |
| tier: fast / network / orekit / slow / gmat-ToDo (GMAT checks are listed only, not tested) |
Then: list any contigo API the notebook uses that no longer exists, and say
whether the notebook is FULLY covered by §4 (delete) or not (archive), with a
one-line reason. Do not edit files. Output only the tables, under 150 lines.
```

The main session merges the three reports into §4 ("Source" column and NEW rows) and §5.1. **Kyle reviews the updated catalogue before any test is written.**

### 6.3 Step B: write the tests (general-purpose, worktree, parallel)

1. Serial first: **T0** (conftest + fixtures + `tests/data/make_fixtures.py`) is done in the main session and merged, because every test PR depends on it.
2. Then dispatch up to 3 worktree agents at once, with the model from §6.1: e.g. T1 (sonnet) + T3 (sonnet) + T4 (opus), then T2 (sonnet) + T5 (opus) + T6 (opus), and 0.4 (sonnet) alongside whichever slot is free.

Prompt template:

```
Implement PR <T-id> from docs/phase0_plan.md §4.<n> in this worktree (branch
`tests/<short-name>` from up-to-date main). Implement exactly the rows <IDs>,
using the fixtures in tests/conftest.py (do not change conftest; if you need a
new fixture, put it in a local conftest in your test folder and say so).
Rules:
- Tests only. Do NOT modify anything under contigo/. If a test exposes a bug,
  mark it pytest.mark.xfail(strict=True, reason="<finding id>: ...") and report it.
- Unmarked tests must not touch the network or start the JVM; mark those
  @pytest.mark.network / orekit / slow. Write NO tests that call GMAT (P10).
- Follow docs/plotting_plan.md §7 style: module docstring with
  "added: DD/MM/YYYY Claude Code", numpy docstrings, black, ruff clean.
- Physics tests must state the expected value and where it comes from.
- Run: black tests && ruff check tests && pytest tests/<path> -q
  (and with --run-all if the tool is available locally). Commit locally in the
  worktree with a clear message; do not push.
Report: files changed, line count, pytest summary, list of xfails with reason,
and any catalogue row you could not implement as written (and why).
```

3. When each agent finishes, the main session reads the diff, runs `/code-review medium` on the branch, and fixes issues or sends follow-ups to the same agent (`SendMessage`). Then it hands the branch to Kyle (§6.4).

### 6.4 Hand-off to Kyle before every PR (P8)

For each branch, the main session posts:

1. **What changed:** `git diff --stat main...<branch>`, the key hunks (API changes, any edit
   under `contigo/`, new xfails), and the local pytest/ruff/black results, including which
   marked tiers were run.
2. **Suggested PR text**, ready to paste: title, the summary, the plotting-plan §6 checklist
   filled in, linked catalogue/refactor IDs, xfails added or removed, and anything Kyle should
   check by hand.
3. **The commands** to push the branch (`git push -u origin <branch>`) and the compare URL.

Kyle reviews, pushes and opens the PR. Review comments come back to the main session, which
updates the branch locally and posts a new hand-off.

### 6.5 Step C: refactors (one at a time per area)

- For R2, R3+R5 and R11: first dispatch a `Plan` agent (`opus`) with "design the minimal change for R-n; list files, API changes, risks, which xfails flip". Kyle approves the design.
- Then one worktree `general-purpose` agent per refactor, on the model from §6.1, with the prompt: "Implement R-n exactly as approved; remove the matching xfail markers; all fast-tier tests must stay green; do not touch unrelated files."
- Refactors that touch the same file (`orekit_utils.py`: R5, R11) go to the **same** agent, one after the other.

### 6.6 Coordination rules

- One task = one branch = one PR; worktree branches always start from the latest `main`. Because Kyle merges, the next dependent branch starts only after the previous PR merges (or is rebased onto it locally, then re-handed-off).
- Kyle's work (0.2/0.3/0.6) starts after all of Claude's Phase 0 PRs merge (P7), so no file freeze is needed during Claude's work.
- When a subagent reports something outside its task (a new bug), the main session adds it to §2 and does not let the agent fix it in that PR.
- The plan is updated in the same PR that changes the decision (keep it the single source of truth).

---

## 7. Resolved questions (Kyle, 2026-11-09)

| # | Question | Answer |
|---|----------|--------|
| 1 | Renumber SPIRE to 0.6; Claude takes 0.1 | Yes (P5, P6) |
| 2 | Python 3.13/3.14 in CI | Yes (P11) |
| 3 | Accept lower-case `out_scale` | Yes (TU-03, R10) |
| 4 | SP3 loading | Fix now (R8) |
| 5 | Orekit as an extra | No: Orekit is core (P9; R14 dropped) |
| 6 | ED-08 tolerance | Keep it loose for now; GMAT likely to be removed, so no GMAT tests (P10, R15) |
| 7 | EP-06 ~1 km | Yes |
| 8 | Deprecation: warn in 0.2.0, remove in 0.3.0 | Yes (R9, and R15 if approved) |

No open questions remain for Phase 0. Next step: §6.2 (notebook mining).
