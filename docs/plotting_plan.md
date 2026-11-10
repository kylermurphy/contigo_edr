# CONTIGO Plotting Module: Work Plan

| | |
|---|---|
| **Status** | v2 (2026-11-10): Phase 0 done, Phase 1 next |
| **Lead / reviewer** | Kyle Murphy (reviews PRs within 24 hours) |
| **Implementer** | Kevin Delano |
| **Branch** | One branch per task (see Section 5 and `CONTRIBUTING.md`) |
| **Scope** | Static (matplotlib) exploratory plots for `Spacecraft` and `Constellation` |
| **Out of scope (later plans)** | Interactive plots (Plotly), EDR/EFD result plots |

> **How to comment on this plan:** open a pull request that edits this file, or
> comment on the PR that adds it. Suggest changes inline. Anything marked
> **[DECISION]** is a choice we made; anything marked **[OPEN]** still needs input.

---

## 1. Goals

Give CONTIGO users a fast way to *look at* their spacecraft data, so that the data can 
be visualized and problems in the data (gaps, wrong units, wrong frame) are seen before 
they reach the EDR/EFD calculation. Phase 1 plots:

1. **Altitude** vs time
2. **Ground track** (geographic map, shows the latitude extent / inclination)
3. **Orbit** (3-D, inertial frame)

Each plot is available two ways, backed by **one** implementation:

```python
from contigo import plotting
plotting.plot_altitude(constellation)     # plotting module
constellation.plot_altitude()             # method wrapper (same result)
spacecraft.plot_altitude()                # works on a single Spacecraft too
```

### Design decisions

| # | Decision | Why |
|---|----------|-----|
| D1 | **[DECISION]** Exploratory, static matplotlib | Fast, works in Colab, no extra servers. Interactive is a second plan. |
| D2 | **[DECISION]** Altitude = WGS-84 geodetic height above the ellipsoid | Matches what `jb08_utils` and density models expect. |
| D3 | **[DECISION]** Orbit plot is inertial (J2000) by default, Earth-fixed optional | An ECEF orbit is a spiral that hides the orbit shape. |
| D4 | **[DECISION]** Geodetic conversion is pure NumPy; the ECEF to inertial transform uses SPICE (`sxform`), **not Orekit** | See benchmark and rationale below. Plotting must work without Java or network access. |
| D5 | **[DECISION]** Basemap is plain matplotlib with a small bundled coastline file; Cartopy optional later | Cartopy is hard to install on some systems and Colab setups. |
| D6 | **[DECISION]** Wrappers are flat methods (`sc.plot_altitude()`) | Simplest to learn. Revisit an accessor (`sc.plot.altitude()`) if plot count grows. |
| D7 | **[DECISION]** Plan lives in `docs/` in the repo; docs site (`mkdocs`) comes later | Reviewable via PR; the implementer does not need other tools. |
| D8 | **[DECISION]** Tests use small synthetic fixtures; the example notebook uses the new SPIRE cubesat dataset | Tests must not depend on large or yet-to-exist data. |

### Transform benchmark (supports D4)

Measured on the merged Phase 0 code (single core, NumPy + SpiceyPy, kernels from `contigo/data/`):

| Operation | Result |
|-----------|--------|
| `sxform('ITRF93','J2000', et)` in a Python loop | ~5 µs per epoch (20,000 epochs in 0.10 s) |
| Vectorised ECEF to geodetic (pure NumPy, 5 iterations) | 1,000,000 points in ~0.2 s; round-trip error < 1e-7 m for altitudes up to 40,000 km |
| `ecef_to_inertial` against Orekit's ECI state (`orekit_ref_small`, two orbits) | max 0.81 m, 0.9 mm/s |
| Fallback `IAU_EARTH` against `ITRF93` (LEO) | 8 to 10 km, 9 to 12 m/s. Only used when the Earth orientation kernel does not cover the dates |
| Earth-rotation-angle-only ECEF to inertial (no SPICE) | **~34 km** position error. Rejected: it ignores precession and nutation, which tilts inclination by tenths of a degree. |

Conclusion: **decimate first, then transform with SPICE.** A plot never needs more
than a few thousand points per spacecraft, so the SPICE loop costs well under a
second even for hundreds of spacecraft. The geodetic conversion is fast enough to
run on the full data when needed (altitude statistics), and decimation makes the
plot cheap either way.

### Why SPICE and not Orekit for the plotting transforms (D4)

SPICE is already a hard dependency of the core: `Spacecraft` calls `spiceypy` for every time
conversion (`sspice_et`, `sc_utc`) and `time_utils` loads the leap-second kernel. What is
"less used" is `SPICEEphem` (the solar-system *ephemeris* provider). The plotting transform needs
only the leap-second kernel and the Earth orientation kernel, which is a rotation, not an ephemeris.

Orekit's strength here (ephemeris caching) does not help plotting: a plot needs one transform
per decimated epoch, with nothing to reuse. Against it:

- `start_orekit()` starts a JVM, queries the GitLab API for the latest orekit-data commit, and may
  download data. A plot should not need Java or network access (think Colab cold starts).
- Efficient use needs a new Java batch helper in `java_src/` (like `EphemerisBatchHelper`) and a
  rebuilt, committed jar. That is a heavy ask for a first-time contributor.
- Per-point calls through JPype are slow without such a helper.

Keep the transform in **one function** (`ecef_to_inertial`) so an Orekit backend can be added later
if EDR code needs frame consistency with Orekit's ITRF. No backend switch now.

**Kernel coverage:** the bundled `earth_latest_high_prec.bpc` covers 2000-01-01 to
**2026-05-30**. `kernels.check_pck` (called by `ecef_to_inertial`) checks coverage with
`spice.pckcov`, tries one update from NAIF into the user cache per session, and otherwise warns
(`RuntimeWarning`) and falls back to the low-precision `IAU_EARTH` frame, which needs the bundled
text kernel `pck00011.tpc`. The fallback is off by ~10 km at LEO (about 0.08 degrees of longitude):
fine for a plot, not for EDR (`fallback="raise"`). With `CONTIGO_OFFLINE=1` nothing is downloaded.
Orekit has the same limit through its EOP data.

### Performance

"Hundreds of spacecraft" is fine if every plot function:

- decimates to `max_points` per spacecraft **(default 2000) before** any transform,
- does the math vectorised over `(N, 3)` arrays, never a Python loop over points
  (the one exception is the per-epoch SPICE call, which is cheap after decimation),
- draws one `Line2D` per spacecraft (or a `LineCollection`), never one per point.

---

## 2. From the existing codebase

These shape the design. Please read before starting.

- `Spacecraft.state_ecef` is `(N, 6)` `[x, y, z, vx, vy, vz]` in metres and m/s, Earth-fixed.
  `Spacecraft` itself has no altitude, geodetic lat/lon or orbital element; `contigo_utils.coords`
  computes the first two (and the inertial position), orbital elements are Phase 3.
- `Spacecraft.spherical()` returns `[r, lat, lon]` with **geocentric** latitude in radians and
  `r` from Earth's centre. `r` is not altitude. Do not use it for the altitude plot.
- Time is available as `stime` (pandas `DatetimeIndex`), `sspice_et` (TDB seconds past J2000),
  `sspice_gps` and `sc_utc`. Use `stime` or `sc_utc` for plot axes, `sspice_et` for SPICE calls.
- `Constellation` is a dict of single-ID `Spacecraft` (`constellation.spacecraft[id]`), iterable
  (`for sc in constellation`), with `.ids`. A single `Spacecraft` can contain several IDs until
  `split_by_id()` is called; plotting code should always go through `split_by_id()` or a
  `Constellation` so each line is one spacecraft.
- SPICE kernels are handled by `contigo_utils/kernels.py` (cache first, then bundled `contigo/data`;
  downloads go to the user cache). `coords.ecef_to_inertial` loads what it needs itself through
  `kernels.check_pck`, so no `SPICEEphem` is required and plotting code never loads kernels.
- Packaging (`packages.find`), the `plot` extra and the test suite are in place. matplotlib is an
  optional dependency, imported lazily (`contigo.plotting._style.require_matplotlib`).

---

## 3. Architecture

```
contigo/
  contigo_utils/
    coords.py         # DONE: ecef_to_geodetic, geodetic_to_ecef, ecef_to_inertial, inertial_to_ecef
    kernels.py        # DONE: SPICE kernel lookup/download/load, check_pck
  plotting/
    __init__.py       # DONE (empty __all__); public API: plot_altitude, plot_ground_track, plot_orbit
    _style.py         # DONE: require_matplotlib, color_for, color_map, rc_context
    _util.py          # DONE: as_spacecraft_dict, decimate, split_antimeridian
    altitude.py
    ground_track.py
    orbit.py
    data/coastline.npz  # small bundled coastline (lon, lat, NaN-separated)
tests/
  test_coords.py, test_earth_orientation.py   # DONE
  plotting/
    conftest.py, test_util.py, test_style.py  # DONE
    test_altitude.py ...
notebooks/
  tutorials/Plotting_Examples.ipynb    # Colab-friendly gallery
docs/
  plotting_plan.md           # this file
  coordinates.md, testing.md, ToDo.md
```

This mirrors the physics layer: pure maths in `contigo_utils/coords.py` (like `grav_utils.py`), drawing code in
the plot modules. The maths can be tested without ever making a figure.

### Public API conventions (every plot function)

```python
def plot_<name>(
    data: Spacecraft | Constellation,
    *,
    ax: Axes | None = None,        # draw on a user-supplied axes, else create one
    max_points: int = 2000,        # decimation per spacecraft
    ids: Sequence | None = None,   # subset of spacecraft IDs
    **plot_kwargs,                 # forwarded to ax.plot (color, lw, ...)
) -> tuple[Figure, Axes]:
```

Rules:

1. Accepts a `Spacecraft` **or** a `Constellation`.
2. Returns `(fig, ax)`. Never calls `plt.show()` and never calls `plt.figure()` globally if `ax` is passed.
3. Does not change global matplotlib state (use `with plt.rc_context(...)`).
4. Same spacecraft ID gets the same colour in every plot (`_style.color_for(id)`).
5. Units internally are metres; **axes are labelled in km** and say so.
6. Raises `ValueError` with a helpful message for bad input (follow the style of `jb08_utils._validate_geodetic`).

### Wrapper pattern (one per class, per plot)

Import **inside** the method to avoid a circular import (`plotting` imports `Spacecraft` for typing).

```python
# in spacecraft.py and constellation.py
def plot_altitude(self, **kwargs):
    """Plot geodetic altitude vs time. See :func:`contigo.plotting.plot_altitude`."""
    from contigo.plotting import plot_altitude
    return plot_altitude(self, **kwargs)
```

---

## 4. Work phases

Each task lists **acceptance criteria**. A task is done when all of them are true **and** the PR
checklist in Section 6 passes. One PR per task unless noted.

### Phase 0: Foundations (done)

Everything the plots build on is merged (lead). The coordinate and kernel functions are documented
in [`coordinates.md`](coordinates.md); the test suite (tiers, fixtures, test IDs) in
[`testing.md`](testing.md). Open follow-ups are in [`ToDo.md`](ToDo.md).

| Task | What landed | Where |
|------|-------------|-------|
| 0.1 | Packaging with `packages.find` (`include = ["contigo*"]`), `plot` extra, pytest suite | `pyproject.toml`, `tests/` |
| 0.2 | `ecef_to_geodetic` (vectorised) and `geodetic_to_ecef`; `WGS84_A`, `WGS84_F`, `WGS84_E2` | `contigo_utils/coords.py`, `constants.py` |
| 0.3 | `ecef_to_inertial` / `inertial_to_ecef` (SPICE ITRF93 to J2000, positions or full states); `kernels.check_pck` with coverage check, one update per session, `IAU_EARTH` fallback with a warning | `coords.py`, `kernels.py`, `data/pck00011.tpc` |
| 0.4 | Plotting skeleton: `as_spacecraft_dict`, `decimate`, `split_antimeridian`, `color_for`, `color_map`, `rc_context`, `require_matplotlib`; `import contigo.plotting` works without matplotlib | `contigo/plotting/`, `tests/plotting/` |
| 0.5 | CI: `ruff` and `black` lint, `pytest` fast tier on Python 3.10, 3.12, 3.13 and 3.14 with a non-editable install | `.github/workflows/ci.yml` |

**Still open from Phase 0:** the new SPIRE ECEF/GPS dataset (Kyle). Until it exists use
`data/ESA_pod.hdf`; it loads with pandas into `Spacecraft` or `Constellation`.

#### Using the Phase 0 code in a plot

```python
from contigo.contigo_utils.coords import ecef_to_geodetic, ecef_to_inertial
from contigo.plotting import _util

idx = _util.decimate(sc.N, max_points)                  # decimate first
alt_km = ecef_to_geodetic(sc.state_ecef[idx, :3])[:, 2] / 1e3
xyz_i = ecef_to_inertial(sc.state_ecef[idx, :3], sc.sspice_et[idx])   # (N, 3) in, (N, 3) out
```

| Plot | Needs | From |
|------|-------|------|
| Altitude | geodetic height | `ecef_to_geodetic(r)[:, 2] / 1e3` (km); no SPICE |
| Ground track | geodetic latitude and longitude (radians) | `ecef_to_geodetic(r)[:, :2]`, then `_util.split_antimeridian(lon, lat)`; convert to degrees at the plotting boundary |
| Orbit, `frame="inertial"` (default) | J2000 positions | `ecef_to_inertial(r, et)` with `r = sc.state_ecef[idx, :3]` and `et = sc.sspice_et[idx]` |
| Orbit, `frame="ecef"` | Earth-fixed positions | `sc.state_ecef[idx, :3]` as is, no transform |
| Elements (task 3.3) | J2000 position and velocity | `ecef_to_inertial` on the `(N, 6)` state; the velocity includes Earth's rotation |

Rules for plotting code:

- **Decimate before transforming.** The transform is a Python loop of about 5 µs per epoch.
  `decimate` keeps the first and last sample and returns at most `max_points` indices.
- `et` is `sspice_et` (TDB seconds past J2000) at the same indices as the positions.
- Positions in, positions out: an `(N, 3)` input returns `(N, 3)`. An `(N, 6)` state returns
  `(N, 6)`.
- **Dates after 2026-05-30** may not have Earth orientation data. The function then warns with a
  `RuntimeWarning` and is off by ~10 km. Plots can leave the warning visible (do not suppress it);
  EDR code passes `fallback="raise"`.
- **Pick example and test dates before 2026-05-30** (inside the bundled coverage). Then the notebook
  and the tests work offline, on CI and on a Colab cold start, with no download and no warning. This
  applies to the SPIRE example data too. Tests that need an uncovered date set `CONTIGO_OFFLINE=1`
  and expect the warning (as `tests/test_coords.py` FX-06 does).
- Nothing needs Java or the network when the dates are covered (and with `CONTIGO_OFFLINE=1`).

### Phase 1: Altitude plot (implementer close review)

This is the **template** for every later plot. Take your time, review it, then repeat.

The smoke test is a quick, shallow test that checks the code runs at all without crashing. 
It doesn’t check that the answer is right. A smoke test of plot_altitude would build a small 
synthetic constellation, call the function, and check that it returned a figure with one line 
per spacecraft. It would catch a typo, a wrong import, or a crash on a Constellation input. 
It would not catch an altitude that’s off by 6,378 km or in the wrong units, because it 
never checks the numbers.

I've (Kyle) always found tests difficult with research code. I've found Claude helpful for
suggesting simple tests. 

```python
def test_plot_altitude_runs(constellation):
    fig, ax = plot_altitude(constellation)
    assert len(ax.lines) == len(constellation)
```

| Task | Description | Acceptance criteria |
|------|-------------|---------------------|
| 1.1 | `plotting/altitude.py` with `plot_altitude` (alt vs time, one line per spacecraft, km) | Works on a single `Spacecraft` and a `Constellation`; returns `(fig, ax)`; legend lists IDs; x-axis is dates |
| 1.2 | Wrappers on `Spacecraft` and `Constellation` | Both call the same function; docstring links to it |
| 1.3 | Tests: pure-data test (known ECEF to expected altitude), smoke test (figure created, number of lines = number of IDs) | `pytest tests/plotting` green |
| 1.4 | Example cells in `notebooks/tutorials/Plotting_Examples.ipynb` (with an "Open in Colab" badge) | Notebook runs top to bottom on the SPIRE cubesat dataset (**dependency:** Kyle supplies it; until then use `data/ESA_pod.hdf`). **Clear the outputs before committing** (`jupyter nbconvert --clear-output --inplace notebooks/tutorials/Plotting_Examples.ipynb`): `tests/test_notebooks_size.py` fails any tutorial over 1 MB, and saved plot images pass that quickly. `data/` is not part of the installed package, so in Colab the notebook downloads its data file from GitHub, as the other tutorials do |

### Phase 2: Ground track and orbit plots

| Task | Description | Acceptance criteria |
|------|-------------|---------------------|
| 2.1 | `plot_ground_track`: lon/lat (from `ecef_to_geodetic`) on a plain map with bundled coastline, lines **broken at the antimeridian** (`_util.split_antimeridian`), optional `color_by=` (time or altitude) | No horizontal streaks across the map; lat limits visible so inclination can be read; ground track of an ~87 deg inclination spacecraft (CHAMP, bundled as `contigo/data/CHAMP_POD_20050816_20050822.hdf`) visibly reaches ±87; the coastline file ships in the installed package (`package-data` and the CI install check, see [`ToDo.md`](ToDo.md)). **Andreas had a really cool version of this plot in his last presentation**. |
| 2.2 | Wrappers + tests + notebook cells | As Phase 1 |
| 2.3 | `plot_orbit`: 3-D axes, Earth sphere (or oblate ellipsoid), orbit lines in J2000 (from `ecef_to_inertial`), `frame='inertial'` (default) or `'ecef'` | Equal-aspect 3-D axes; a circular orbit plots as a circle; `frame='ecef'` shows the spiral |
| 2.4 | Wrappers + tests + notebook cells | As Phase 1 |

### Phase 3: Suggested additions (pick in priority order)

| Task | Plot | Why it matters |
|------|------|----------------|
| 3.1 | `plot_coverage`: timeline bar per spacecraft with gaps | Instant data-quality check for multi-satellite data, kind of like a Gantt chart. |
| 3.2 | `plot_lst_lat`: local solar time vs latitude | Standard way to show density-sampling coverage |
| 3.3 | `plot_elements`: a, e, i, RAAN vs time (osculating, from the inertial state) | The proper inclination-vs-time plot |
| 3.4 | `plot_separation`: relative distance between spacecraft pairs | Constellation geometry |

### Phase 4: Documentation and polish

README plotting section, docstring review, example gallery notebook, optional `mkdocs` page.

### Later plans (not in this plan)

- Interactive plots (Plotly) and a plot dashboard, second plan.
- Plots of `EDRDensity` results (EDR/EFD time series and maps), another phase.

---

## 5. Getting started: workflow for a new contributor

The setup and the workflow for every task are in [`CONTRIBUTING.md`](../CONTRIBUTING.md): getting
the repository (command line, GitHub Desktop, or the browser for small edits), the Python
environment, the branch / commit / check / push / PR steps with their GitHub Desktop equivalents,
and the working rules. Read it before starting Phase 1.

For plotting tasks:

- Name branches `plot/<task>`, for example `plot/altitude` or `plot/ground-track`.
- One task from Section 4 per branch and per PR.
- Use the plotting checklist in Section 6 (it extends the general one in `CONTRIBUTING.md`).
- Branch protection on `main` is on: PRs need one approval and green `lint` and `test (3.x)`
  checks.

---

## 6. Pull request checklist

Copy this into every PR description.

```
- [ ] Task id from docs/plotting_plan.md: ____
- [ ] File starts with a docstring and `added: DD/MM/YYYY Name <email>` line
- [ ] Public functions have numpy-style docstrings with Parameters / Returns / Raises
- [ ] Type hints on function signatures (numpy.typing for arrays)
- [ ] No plt.show(); returns (fig, ax); accepts ax=
- [ ] Units: data in metres, axes labelled in km with the unit shown
- [ ] black, ruff, and pytest pass locally
- [ ] Wrapper added on BOTH Spacecraft and Constellation (plot tasks only)
- [ ] Example added to notebooks/tutorials/Plotting_Examples.ipynb, run top to bottom
- [ ] Screenshot of the plot attached to the PR
```

---

## 7. Code style and templates

Follow the existing code (see `spacecraft.py`, `forces/base.py`, `contigo_utils/jb08_utils.py`).

- Module docstring with `added:` line, `from contigo...` absolute imports, `logging.getLogger(__name__)`.
- numpy-style docstrings (newer files use these; older ones mix in Google style, so don't copy those).
- Line length 88, `snake_case`, private helpers start with `_`.
- Import matplotlib **inside** plotting modules, not in `contigo/__init__.py`.

### 7.1 Pure function (testable without a figure)

The reference is now real code: `contigo/contigo_utils/coords.py` (`ecef_to_geodetic`,
`geodetic_to_ecef`) with its tests in `tests/test_coords.py`. Copy its pattern for any new maths:
module docstring with an `added:` line, `from __future__ import annotations`, numpy-style
docstring with Parameters / Returns / Raises, a `ValueError` naming the bad shape, and the tests
stating where each expected number comes from. The geodetic helpers take and return `(..., 3)`
arrays in radians and metres.

### 7.2 Frame transform (inertial)

`ecef_to_inertial(state_ecef, et, *, fallback="warn")` in `coords.py`, with the inverse
`inertial_to_ecef`. It builds one `sxform('ITRF93', 'J2000', t)` per epoch and applies it to the
`(N, 6)` states (or `pxform` for `(N, 3)` positions). `sxform` is the 6x6 state transform, so
it includes Earth's rotation in the velocity (`v_inertial = R (v_ecef + omega x r)`). It is a
Python loop, so always decimate first. `kernels.check_pck(et)` makes sure the kernels are
loaded and returns the frame to use (`"ITRF93"` or the fallback `"IAU_EARTH"`).

### 7.3 A plot function skeleton

```python
def plot_altitude(data, *, ax=None, max_points=2000, ids=None, **plot_kwargs):
    from contigo.contigo_utils.coords import ecef_to_geodetic
    from contigo.plotting import _style, _util

    plt = _style.require_matplotlib()              # clear ImportError if missing
    sc_dict = _util.as_spacecraft_dict(data, ids)  # {id: Spacecraft}
    colors = _style.color_map(sc_dict)             # one colour per ID, clashes resolved

    with _style.rc_context():                      # shared style, global rcParams restored
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.figure

        for sc_id, sc in sc_dict.items():
            idx = _util.decimate(sc.N, max_points)  # indices, evenly spaced
            alt_km = ecef_to_geodetic(sc.state_ecef[idx, :3])[:, 2] / 1e3
            ax.plot(sc.stime[idx], alt_km, color=colors[sc_id],
                    label=str(sc_id), **plot_kwargs)

        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Geodetic altitude (km)")
        ax.legend()
    return fig, ax
```

(Add a "no legend if more than ~15 spacecraft" guard so hundreds of lines don't produce a giant legend.)

### 7.4 Test templates

```python
# the numbers going into the plot are already tested in tests/test_coords.py
# (e.g. test_cd01_equator_pole_and_altitude); a plot test only needs the figure:

# tests/plotting/test_altitude.py  (smoke test)
import matplotlib
matplotlib.use("Agg")           # no display needed

def test_plot_altitude_one_line_per_spacecraft(constellation):
    fig, ax = plot_altitude(constellation)
    assert len(ax.lines) == len(constellation.spacecraft)
```

Use the fixtures in `tests/plotting/conftest.py` (`multi_sc`, a three-spacecraft `Spacecraft`, and
`constellation`) rather than loading the large HDF files.

### 7.5 Gotchas to expect

- **Antimeridian:** a ground track jumps from +180 to -180; without breaking the line a horizontal
  streak crosses the map. Use `_util.split_antimeridian(lon, lat)`, which inserts `NaN` breaks
  where `abs(np.diff(lon)) > pi`.
- **`sc_id` is a pandas category array**, so `np.unique` / dict keys work but be careful when comparing to strings.
- **Datetime axes:** `stime` is a pandas `DatetimeIndex`; matplotlib handles it directly.
- **3-D equal aspect:** `ax.set_box_aspect((1, 1, 1))`; a flat-looking Earth means this is missing.
- **Degrees vs radians:** transforms return radians. Convert once, at the plotting boundary.
- **Large `N`:** decimate with `_util.decimate(sc.N, max_points)` (evenly spaced, unique indices
  that keep the first and last sample) before transforms, not after.

---

## 8. References

- matplotlib "Quick start" (figures, axes, artists): <https://matplotlib.org/stable/users/explain/quick_start.html>
- matplotlib 3-D plotting: <https://matplotlib.org/stable/gallery/mplot3d/index.html>
- pytest "Get started": <https://docs.pytest.org/en/stable/getting-started.html>
- SpiceyPy `sxform` / `pxform`: <https://spiceypy.readthedocs.io/>; NAIF frames required reading: <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html>
- WGS-84 geodetic conversion background: Bowring (1976) / Heikkinen (1982) closed-form methods.
- Git basics for scientists: <https://swcarpentry.github.io/git-novice/>
- numpy docstring style: <https://numpydoc.readthedocs.io/en/latest/format.html>

---

## 9. Risks and open items

| ID | Item | Mitigation / question |
|----|------|-----------------------|
| R1 | ~~SPICE kernels must be loaded for the inertial transform~~ | **Resolved:** `kernels.check_pck` loads them (task 0.3) |
| R2 | Plots with hundreds of spacecraft are visually unreadable | Default legend off above ~15 spacecraft; `ids=` subset; colormap by index |
| R3 | `Spacecraft` with several IDs and unsorted time | Always go through `split_by_id()` (sorts per ID) |
| R4 | Bundled Earth PCK ends 2026-05-30 (checked with `pckcov`); newer data needs a download or the fallback | **Handled in `check_pck`:** one NAIF update per session into the user cache, else `RuntimeWarning` and `IAU_EARTH` (~10 km). Plots show the warning; EDR uses `fallback="raise"` |
| R5 | Bundled coastline licence | Use Natural Earth (public domain), downsample to under ~100 KB |
| O1 | ~~Reviewer / turnaround~~ | **Resolved:** Kyle, within 24 hours |
| O2 | ~~CI~~ | **Resolved:** ruff, black and pytest (task 0.5); required checks `lint` and `test (3.x)` |
| O3 | ~~Example dataset~~ | **Resolved:** new SPIRE cubesat dataset (Kyle to create); synthetic fixtures for tests |
| O4 | ~~Docs site~~ | **Resolved:** later |
| O5 | ~~Which contributor takes Phase 1?~~ | **Resolved:** Kevin takes Phase 1; Phase 0 has landed |
| O6 | ~~Branch protection on `main`~~ | **Resolved:** protection is on. Kyle (repo owner) can bypass it and push to `main`; other contributors cannot and must open PRs |
| O7 | **[OPEN]** SPIRE ECEF/GPS dataset (Kyle) | Needed for the notebook cells (1.4); `data/ESA_pod.hdf` until then |
