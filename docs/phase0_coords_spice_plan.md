# CONTIGO Phase 0: tasks 0.2 and 0.3 (geodetic, frames, Earth orientation kernels)

| | |
|---|---|
| **Status** | Draft v2 (2026-11-10), rebased on the merged Phase 0 work |
| **Parent plans** | [`plotting_plan.md`](plotting_plan.md) tasks 0.2 and 0.3; [`phase0_plan.md`](phase0_plan.md) §1 |
| **Owner** | Kyle Murphy |
| **Based on** | `main` at `e067932` (PR #36). v1 was written against `0d6cfa9` |
| **Verified** | The code below was run on `e067932` (Python 3.13, SPICE N0067). §7 lists what was and was not run |

> Markers as in `phase0_plan.md`: **[DECISION]** = agreed, **[PROPOSED]** = needs Kyle's OK,
> **[OPEN]** = needs input.

---

## What changed since v1

22 commits landed. One of them reshapes this plan:

**PR #33 (R16b) already built the shared kernel layer.** `contigo/contigo_utils/kernels.py` now provides
`naif_url`, `ensure_kernel` (cache then bundled lookup via `config.find_data_file`, download only into
`cache_dir("spice")`, optional NAIF freshness check, offline-aware), `furnsh_kernel` (load once, compared by
normalised real path, unloads a same-name kernel at another path first) and `is_loaded`. `check_lpsk` and
`SPICEEphem` already call it, TU-05 was already updated, and `tests/test_kernel_paths.py` covers it.

Effect on v1:

| v1 item | Now |
|---------|-----|
| New `spice_utils.py` with `locate_kernel`, `furnsh_once`, `fetch_kernel`, `ensure_kernel` | **Dropped.** `kernels.py` is that module. Only the Earth-orientation part is new, and it goes **into `kernels.py`** |
| Refactor of `check_lpsk` and `SPICEEphem`, TU-05 update, SU-01 to SU-07 | **Dropped.** Done by R16b |
| Decision S1 (extract and reuse), S2 (downloads to the cache) | Met by R16b. Not our work any more |
| PR-2 of about 450 lines | Now about 260 lines (119 of them in `kernels.py`) |
| Fallback needs a text PCK (F28), PCK ends 2026-05-30 (F30) | Unchanged, still true on the new `main` |

Other merged work checked for conflicts: `config.py` (docstring only), `utils.dl_file` (timeouts, returns a size; our code never relied on its old signature), `tests/_support.py` and `conftest.py` (now use `kernels.furnsh_kernel`; no impact), `time_utils.spice_time` (lower-case scales; our tests use upper-case). The `0.2`/`0.3` rows of `plotting_plan.md` are untouched.

---

## 0. Decisions

| # | Decision |
|---|----------|
| S3 | **[DECISION]** Out-of-coverage default is warn + `IAU_EARTH`. `ecef_to_inertial(..., fallback="warn" \| "raise")` lets EDR code be strict. |
| S4 | **[DECISION]** This plan is a markdown file in `docs/`; Kyle reviews and pushes (P8). |
| P1 | **[DECISION]** (was proposed) `check_pck` is not in `time_utils.py` as the plotting plan says. It lives in **`kernels.py`** next to `ensure_kernel`/`furnsh_kernel`, which it uses. `time_utils` and `coords` import it from there. |
| P2 | **[DECISION]** New `config.state["pck_refreshed"]`: a failed or offline update must not be retried on every call. `check_pck` also skips its update if `SPICEEphem` already did the session's NAIF check (`kernel_downloaded`). |
| P3 | **[DECISION]** Fallback notice is `warnings.warn(..., RuntimeWarning)` plus `logger.info`. |
| P4, P5 | **[DECISION]** `geodetic_to_ecef` and `inertial_to_ecef` are added; `ecef_to_inertial` also accepts (N, 3) positions. |
| P6 | **[DECISION]** Bundle `pck00011.tpc` in `contigo/data/` (listed in `package-data` and in the CI install check). **Your step:** download the real file (§2). My tests ran with a stand-in holding only the Earth constants. |

---

## 1. Findings that still apply

| ID | Finding | Action |
|----|---------|--------|
| F28 | **`IAU_EARTH` cannot be used with the bundled kernels.** `pxform('IAU_EARTH', 'J2000', et)` raises `SPICE(FRAMEDATANOTFOUND)`: it needs a *text* PCK (`pck0001x.tpc`) for the pole and rotation constants; only `.tls`, `.bsp` and `.bpc` are bundled. | P6 |
| F29 | **The fallback is ~10 km, not "of order km".** With a stand-in text kernel (P6), `IAU_EARTH` vs `ITRF93` differs by 8.1 km / 8.9 m/s (2018 ESA orbit vs Orekit ECI) and 10.3 km (2026-05-29). About 0.08° of longitude at LEO: fine for a plot, wrong for EDR. | docstrings; fix plotting-plan R4 row |
| F30 | **Today's data is already outside the bundled PCK.** `earth_latest_high_prec.bpc` covers ITRF93 (body 3000) from 2000-01-01 to **2026-05-30**. `pckcov` + `wnincd` give the check; an uncovered `sxform` raises `SpiceFRAMEDATANOTFOUND`. | `check_pck` |
| F32 | `kernels._download` unloads a file, then `dl_file` writes straight to the target. A cut-off transfer leaves a truncated kernel in the cache, which is found as "local" and fails `furnsh` on every later run. This matters more now that the PCK is refreshed when NAIF republishes it. | **[OPEN] Q1** (follow-up to R16b, not part of these PRs) |
| F34 | `ensure_kernel` uses `config.find_data_file`, which looks in the **current directory first**. A stray `naif0012.tls` or `earth_latest_high_prec.bpc` in the working directory would silently win. | Note only; R16b's design |

Resolved by R16b (v1 F31, F33): duplicate kernel logic and inconsistent path comparison.

Plotting-plan wording to fix when PR-2 lands: §1 "Kernel coverage warning" and risk R4 (the fallback needs a text PCK and is ~10 km), and the 0.3 row (`check_pck` is in `kernels.py`).

---

## 2. Work breakdown and how to apply it

| File | Change | PR |
|------|--------|----|
| `contigo/contigo_utils/constants.py` | add `WGS84_A`, `WGS84_F`, `WGS84_E2` | 1 |
| `contigo/contigo_utils/coords.py` | **new**: `ecef_to_geodetic`, `geodetic_to_ecef` | 1 |
| `tests/test_coords.py` | **new**: CD-01 to CD-05 | 1 |
| `contigo/contigo_utils/kernels.py` | add `pck_coverage`, `pck_covers`, `check_pck`, constants | 2 |
| `contigo/config.py` | `TPC_FILE = "pck00011.tpc"`; `state["pck_refreshed"]` | 2 |
| `contigo/data/pck00011.tpc` | **new, you add it** | 2 |
| `pyproject.toml`, `.github/workflows/ci.yml` | `data/*.tpc` in package-data; `pck00011.tpc` in the install check | 2 |
| `tests/test_earth_orientation.py` | **new**: EO-01 to EO-07 | 2 |
| `contigo/contigo_utils/coords.py`, `tests/test_coords.py` | add `ecef_to_inertial`, `inertial_to_ecef`; FX-01 to FX-08 | 3 |

```
PR-1  feat/coords-geodetic   task 0.2   pr1_coords_geodetic.patch    +178 lines   independent
PR-2  feat/earth-orientation task 0.3a  pr2_earth_orientation.patch   +263 -1      independent; needs the real pck00011.tpc
PR-3  feat/ecef-inertial     task 0.3b  pr3_ecef_inertial.patch      +186 -5      after PR-1 and PR-2 are merged
```

### Applying a patch (PR-1 shown; same for the others)

The patches are plain `git diff` output, so nothing is committed for you. From your clone, with the patch files saved in a folder (called `patches` here):

```bash
git fetch origin
git switch -c feat/coords-geodetic origin/main      # new branch from the current main
git apply --check ../patches/pr1_coords_geodetic.patch   # dry run: prints nothing if it fits
git apply --index ../patches/pr1_coords_geodetic.patch    # applies and stages the changes
git status                                          # expect 3 files, all staged
ruff check contigo tests && black --check contigo tests
pytest tests/test_coords.py                         # PR-1: 6 tests
git commit -m "Add WGS-84 ECEF/geodetic conversion (plotting task 0.2)"
git push -u origin feat/coords-geodetic
```

If `git apply` complains about whitespace or line endings (possible on Windows with `core.autocrlf`), add `--ignore-whitespace`; if it reports a conflict because `main` moved, use `--3way` instead of `--index` (it merges where it can and marks the rest).

For PR-2, before running the tests:

```bash
git switch -c feat/earth-orientation origin/main
git apply --index ../patches/pr2_earth_orientation.patch
# PowerShell:
Invoke-WebRequest https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc -OutFile contigo/data/pck00011.tpc
# bash / Git Bash:
curl -o contigo/data/pck00011.tpc https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc
git add contigo/data/pck00011.tpc
pytest tests/test_earth_orientation.py tests/test_kernel_paths.py tests/test_time_utils.py
```

(I could not reach NAIF from my sandbox, so please check that URL resolves.) EO-03, EO-06 and EO-07 fail until the file is in place; that is expected. After the real file is in, also run `pytest tests/test_coords.py` once PR-3 is applied to re-check FX-06 and FX-08, which are the two tests that depend on `IAU_EARTH` numbers.

PR-3: after PR-1 and PR-2 are merged, `git fetch`, `git switch -c feat/ecef-inertial origin/main`, then apply `pr3_ecef_inertial.patch` the same way. (If you would rather stack the branches before merging, branch PR-3 from the PR-2 branch after applying PR-1 to it.)

---

## 3. Design

### 3.1 What `kernels.py` gains

| Function | Purpose |
|----------|---------|
| `pck_coverage(body=3000)` | union of the coverage (ET seconds) of all loaded binary PCKs for the ITRF93 frame |
| `pck_covers(et_min, et_max)` | `wnincd` test on that window |
| `check_pck(et=None, *, fallback="warn")` | make Earth-orientation data for `et` available; returns `"ITRF93"` or `"IAU_EARTH"` |

### 3.2 `check_pck` flow

```
check_pck(et)
 ├─ furnsh leap seconds + Earth PCK            (ensure_kernel(check_remote=False): cache or bundled, no network)
 ├─ et covered by the loaded PCKs?  ──yes──► "ITRF93"
 ├─ kernel_downloaded or pck_refreshed already set?  ──yes──┐   (SPICEEphem or an earlier call already asked NAIF)
 ├─ pck_refreshed = True; ensure_kernel(PCK) with the freshness check
 │     newer NAIF copy → downloaded to cache_dir("spice"), furnsh_kernel unloads the bundled one
 │     offline / HTTP / OS error → INFO log, carry on
 ├─ covered now?  ──yes──► "ITRF93"
 └─ fallback="raise" → RuntimeError naming the covered range
    fallback="warn"  → RuntimeWarning, furnsh pck00011.tpc, "IAU_EARTH"   ◄──┘
```

Whole-call fallback: if only some epochs are uncovered, all use `IAU_EARTH`. Mixing frames would put a ~10 km step into the track at the seam.

### 3.3 Conventions followed

Taken from `CLAUDE.md`, `kernels.py` and the existing tests: module docstring with an `added:` line (`10/11/2026 Claude Code` for files drafted here; change it if you prefer your own name), `from __future__ import annotations`, `import contigo.config as config`, `logger = logging.getLogger(__name__)` with no logging configuration, numpy-style docstrings, SI units and radians, network only through `utils.dl_file`/`utils.wf_mtime` (so `CONTIGO_OFFLINE` and the test guard apply), black (88) and ruff clean.

---

## 4. Test catalogue

Fixtures: `spice_offline` (autouse), `esa_pod_small`, `orekit_ref_small`; `fresh_session` and `spy_ensure` are local to `test_earth_orientation.py`.

### 4.1 Geodetic, `tests/test_coords.py` (PR-1)

| ID | Test | Expected |
|----|------|----------|
| CD-01 | Equator, both poles, +400 km radial | alt 0 to 1e-6 m; lat 0, ±π/2; 400 km to 1e-3 m (WGS-84 a, b = a(1−f)) |
| CD-02 | Round trip, 1e5 random points, alt −100 to 2000 km and 0 to 40,000 km | alt < 1e-6 m (measured 4e-9 and 2e-8 m); lat < 1e-12 rad (measured 1e-15) |
| CD-03 | Last axis not 3 | `ValueError` |
| CD-04 | (4, 5, 3) in | (4, 5, 3) out |
| CD-05 | 1e6 points | < 2 s (measured 0.20 s) |

### 4.2 `tests/test_earth_orientation.py` (PR-2)

| ID | Test | Expected |
|----|------|----------|
| EO-01 | Bundled PCK coverage | covers 2018; does not cover 2026-09 |
| EO-02 | `check_pck` on covered epochs with `wf_mtime`/`dl_file` set to fail | `"ITRF93"`, no network call |
| EO-03 | Uncovered + offline, called 3 times | `RuntimeWarning` each time, `"IAU_EARTH"`, exactly **one** update attempt |
| EO-04 | `fallback="raise"`; bad `fallback` | `RuntimeError` mentioning `2026-05-30`; `ValueError` |
| EO-05 | Online, fake NAIF copy dated 2099, fake download | one download of the PCK, then `"ITRF93"` with warnings turned into errors |
| EO-06 | `kernel_downloaded` already True | no update attempt (SPICEEphem already asked) |
| EO-07 | `IAU_EARTH` rotation with the bundled `pck00011.tpc` | a proper rotation (R·Rᵀ = I, det 1), not the identity |
| EO-08 | `network`: real PCK download into a temp cache | **not written**: furnishes and `pck_covers` reaches past 2026-05-30 |

Teardown note: a test that furnishes a *cached* PCK should `spice.unload` it; the kernel pool is session-wide. EO-05 avoids this by faking `furnsh_kernel`.

### 4.3 Frames, `tests/test_coords.py` (PR-3)

| ID | Test | Expected |
|----|------|----------|
| FX-01 | Equatorial surface point, zero ECEF velocity | inertial speed 465.1 ± 0.1 m/s (measured 465.10) |
| FX-02 | Round trip, 50 random states over a day | < 1e-6 m (measured 4e-9 m, 4e-12 m/s) |
| FX-03 | (N, 3) vs (N, 6) input | positions agree to 1e-9 m |
| FX-04 | Bad width / length mismatch / NaN time | `ValueError` |
| FX-05 | **`esa_pod_small` ECEF → inertial vs `orekit_ref_small` `eci_*`** (fast tier, no JVM) | measured **0.81 m, 0.9 mm/s**; asserted at 5 m and 5 mm/s |
| FX-06 | Epoch after 2026-05-30, offline | `RuntimeWarning` naming `IAU_EARTH`; \|r\| preserved to 1e-9 relative |
| FX-07 | Same, `fallback="raise"` | `RuntimeError` mentioning `2026-05-30` |
| FX-08 | `IAU_EARTH` vs `ITRF93` inside the covered range | < 20 km and < 20 m/s (measured 8 to 10 km, 9 m/s); a guard against a wrong frame, not an accuracy claim |

---

## 5. Template code

Complete files are in the three patches. The parts that carry the design:

### 5.1 `kernels.py`: constants and `check_pck`

```python
#: SPICE body ID that ``pckcov`` reports for the ITRF93 binary PCK
ITRF93_PCK_ID = 3000
#: Earth-fixed frame names returned by :func:`check_pck`
ITRF93 = "ITRF93"
IAU_EARTH = "IAU_EARTH"

#: Errors that mean "could not get a newer kernel"; callers log and carry on
NET_ERRORS = (config.OfflineError, requests.exceptions.RequestException, OSError)
```

```python
def pck_coverage(body: int = ITRF93_PCK_ID) -> SPICEDOUBLE_CELL:
    """Union of the coverage (ET seconds) of all loaded binary PCKs for ``body``."""
    cover = SPICEDOUBLE_CELL(2000)
    for i in range(spice.ktotal("PCK")):
        spice.pckcov(spice.kdata(i, "PCK")[0], body, cover)
    return cover
```

```python
def check_pck(
    et: npt.ArrayLike | None = None,
    *,
    fallback: Literal["warn", "raise"] = "warn",
) -> str:
    """Make Earth orientation data for ``et`` available; return the frame name.

    Furnishes the leap-second and Earth PCK kernels (cache, then bundled; no
    network). If ``et`` is not covered, one attempt per session is made to
    download a newer PCK into ``config.cache_dir("spice")``; this is skipped
    if ``SPICEEphem`` already did its once-per-session NAIF check
    (``config.state["kernel_downloaded"]``) or offline. If ``et`` is still not
    covered, the low-precision ``IAU_EARTH`` frame is returned with a warning,
    or ``RuntimeError`` is raised when ``fallback='raise'``.

    ``IAU_EARTH`` ignores nutation, polar motion and UT1-UTC: expect position
    errors of order 10 km at LEO. That is fine for a plot, not for EDR.

    Parameters
    ----------
    et : array_like, optional
        SPICE ephemeris times (TDB seconds past J2000) that will be
        transformed. If None only the kernels are loaded and ``'ITRF93'`` is
        returned.
    fallback : {'warn', 'raise'}, optional
        Action when ``et`` is not covered, by default ``'warn'``.

    Returns
    -------
    str
        ``'ITRF93'`` or ``'IAU_EARTH'``: the Earth-fixed frame to use.

    Raises
    ------
    RuntimeError
        If ``et`` is not covered and ``fallback='raise'``.
    ValueError
        If ``fallback`` is not ``'warn'`` or ``'raise'``.
    """
    if fallback not in ("warn", "raise"):
        raise ValueError("fallback must be 'warn' or 'raise'")
    for name in (config.LEAP_FILE, config.PCK_FILE):
        furnsh_kernel(ensure_kernel(name, check_remote=False))
    if et is None:
        return ITRF93
    et = np.asarray(et, dtype=float)
    et_min, et_max = float(et.min()), float(et.max())
    if pck_covers(et_min, et_max):
        return ITRF93

    if not (config.state["kernel_downloaded"] or config.state["pck_refreshed"]):
        config.state["pck_refreshed"] = True  # one attempt per session, pass or fail
        try:
            furnsh_kernel(ensure_kernel(config.PCK_FILE))
        except NET_ERRORS as exc:
            logger.info("Could not update %s: %s", config.PCK_FILE, exc)
        if pck_covers(et_min, et_max):
            return ITRF93

    msg = (
        "No Earth orientation data (ITRF93) for the requested epochs; "
        f"{_coverage_text()}."
    )
    if fallback == "raise":
        raise RuntimeError(msg)
    warnings.warn(
        msg + " Using low-precision IAU_EARTH (position error of order 10 km).",
        RuntimeWarning,
        stacklevel=3,
    )
    furnsh_kernel(ensure_kernel(config.TPC_FILE, check_remote=False))
    return IAU_EARTH
```

### 5.2 `coords.py`

`ecef_to_geodetic` is the function in `plotting_plan.md` §7.1 plus input validation.

```python
def geodetic_to_ecef(llh: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert WGS-84 geodetic coordinates to ECEF positions.

    Parameters
    ----------
    llh : array_like
        (..., 3) [lat, lon, alt] with lat/lon in radians and alt in metres.

    Returns
    -------
    numpy.ndarray
        (..., 3) ECEF position [x, y, z] in metres.

    Raises
    ------
    ValueError
        If the last axis of ``llh`` is not 3.
    """
    a = _as_xyz(llh, "llh")
    lat, lon, alt = a[..., 0], a[..., 1], a[..., 2]
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)
```

```python
def _transform(
    state: npt.ArrayLike,
    et: npt.ArrayLike,
    to_inertial: bool,
    fallback: Literal["warn", "raise"],
) -> npt.NDArray[np.float64]:
    """Rotate (N, 3) positions or (N, 6) states between ECEF and J2000."""
    state = np.asarray(state, dtype=float)
    et = np.atleast_1d(np.asarray(et, dtype=float))
    if state.ndim != 2 or state.shape[1] not in (3, 6):
        raise ValueError(f"state must have shape (N, 3) or (N, 6); got {state.shape}.")
    if et.shape != (state.shape[0],):
        raise ValueError(f"et must have shape ({state.shape[0]},); got {et.shape}.")
    if not np.all(np.isfinite(et)):
        raise ValueError("et contains NaN or inf.")

    frame = kernels.check_pck(et, fallback=fallback)
    src, dst = (frame, _INERTIAL_FRAME) if to_inertial else (_INERTIAL_FRAME, frame)
    # sxform includes the Earth-rotation term: v_i = R (v_ecef + omega x r_ecef).
    # It is a Python loop (~5 us per epoch), so decimate before calling.
    func = spice.sxform if state.shape[1] == 6 else spice.pxform
    mats = np.array([func(src, dst, t) for t in et])
    return np.einsum("nij,nj->ni", mats, state)
```

`ecef_to_inertial(state_ecef, et, *, fallback="warn")` and `inertial_to_ecef(...)` call `_transform` with `to_inertial=True` / `False`; the full docstrings are in the patch.

### 5.3 How plotting uses it (PR-3 and later)

```python
from contigo.contigo_utils.coords import ecef_to_geodetic, ecef_to_inertial

idx = _util.decimate(sc.N, max_points)                  # decimate first
alt_km = ecef_to_geodetic(sc.state_ecef[idx, :3])[:, 2] / 1e3
xyz_i = ecef_to_inertial(sc.state_ecef[idx, :3], sc.sspice_et[idx])   # (N, 3) in, (N, 3) out
```

---

## 6. Risks and open items

| ID | Item | Mitigation |
|----|------|-----------|
| Q1 | **[OPEN]** F32: truncated PCK download (see §1) | Suggested follow-up PR: download to `<name>.part`, then `os.replace` in `kernels._download` (about 6 lines; `test_kernel_paths.py` asserts the download target, so its fake would need a small change). Not in these PRs unless you say so |
| O2 | Orekit has the same Earth-orientation limit (plotting plan R4); the two backends diverge for uncovered dates | Out of scope; noted in the docstring |
| R1 | A stale cached PCK older than a newly shipped bundled one wins the lookup | `check_pck` then finds the epochs uncovered and refreshes |
| R3 | `sxform` is a Python loop (~5 µs per epoch measured; 1,104 epochs in 7 ms) | Callers decimate first (already the plotting rule) |
| R4 | Kernel pool and SPICE are process-global and not thread-safe | As today |

---

## 7. What I ran and what I did not

Ran on `e067932` in separate worktrees:

- PR-1 alone: `test_coords.py` (6 tests) pass; ruff and black clean.
- PR-2 alone (with a stand-in `pck00011.tpc`): `test_earth_orientation.py`, `test_kernel_paths.py`, `test_time_utils.py` pass.
- PR-3 on top of 1 and 2: `test_coords.py`, `test_earth_orientation.py`, `test_kernel_paths.py`, `test_time_utils.py`, `test_ephemeris.py`, `test_solar_system_env.py` pass.
- `git apply --check` of PR-1 and PR-2 against `origin/main`: clean. PR-3 applies on the tree after PR-1 and PR-2.

Not run:

- the real `pck00011.tpc` (the `IAU_EARTH` numbers in F29, FX-06, FX-08 and EO-07 used a stand-in);
- anything needing the network (EO-08 is not written; the download path is exercised through fakes);
- Python 3.10/3.12/3.14, Windows, the Orekit tier, and the rest of the suite (my sandbox lacks `pyshtools` and Java).
