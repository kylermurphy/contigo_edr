# CONTIGO Plotting Module: Work Plan

| | |
|---|---|
| **Status** | Draft v1, open for comment |
| **Lead / reviewer** | Kyle Murphy (reviews PRs within 24 hours) |
| **Implementer** | Kevin Delano |
| **Branch** | `contigo-plotting-module-plan` (plan only) |
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

Measured on this repo's environment (single core, NumPy + SpiceyPy, kernels from `contigo/data/`):

| Operation | Result |
|-----------|--------|
| `pxform('ITRF93','J2000', et)` in a Python loop | ~11 µs per epoch (20,000 epochs in 0.22 s) |
| Vectorised ECEF to geodetic (pure NumPy, 5 iterations) | 1,000,000 points in ~0.85 s; round-trip error < 1e-8 m |
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

**Kernel coverage warning:** the bundled `earth_latest_high_prec.bpc` covers 2000-01-01 to
**2026-05-30**. Newer data (e.g. 2026-06 onward) cannot be transformed with it. Task 0.3 therefore
checks coverage (`spice.pckcov`), downloads the current kernel (reusing `SPICEEphem.dl_kernels`
logic), and otherwise warns and falls back to the low-precision `IAU_EARTH` frame (error of order km, fine for
a plot). Orekit has the same limit through its EOP data.

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
  There is **no** altitude, geodetic lat/lon or orbital element anywhere yet.
- `Spacecraft.spherical()` returns `[r, lat, lon]` with **geocentric** latitude in radians and
  `r` from Earth's centre. `r` is not altitude. Do not use it for the altitude plot.
- Time is available as `stime` (pandas `DatetimeIndex`), `sspice_et` (TDB seconds past J2000),
  `sspice_gps` and `sc_utc`. Use `stime` or `sc_utc` for plot axes, `sspice_et` for SPICE calls.
- `Constellation` is a dict of single-ID `Spacecraft` (`constellation.spacecraft[id]`), iterable
  (`for sc in constellation`), with `.ids`. A single `Spacecraft` can contain several IDs until
  `split_by_id()` is called; plotting code should always go through `split_by_id()` or a
  `Constellation` so each line is one spacecraft.
- Kernel loading (`naif0012.tls`, `earth_latest_high_prec.bpc`) currently happens inside
  `SPICEEphem.load_kernels()` and `time_utils.check_lpsk()`. **The frame transform needs the PCK kernel
  loaded even when no `SPICEEphem` exists.** Task 0.3 adds a small shared helper.
- `pyproject.toml` has `packages = ["contigo"]`, which does not include subpackages (`forces/`,
  `ephemeris/`, ...). A normal (non-editable) `pip install` would drop a new `contigo/plotting/`.
  Task 0.1 fixes this.
- matplotlib is not a dependency yet, and there is no `tests/` directory although pytest is configured.

---

## 3. Architecture

```
contigo/
  plotting/
    __init__.py       # public API: plot_altitude, plot_ground_track, plot_orbit
    _style.py         # colour per spacecraft ID, shared rcParams, save helper
    _coords.py        # pure NumPy / SPICE: ecef_to_geodetic, ecef_to_inertial
    _util.py          # as_spacecraft_dict(), decimate(), antimeridian splitting
    altitude.py
    ground_track.py
    orbit.py
    data/coastline.npz  # small bundled coastline (lon, lat, NaN-separated)
tests/
  plotting/
    test_coords.py
    test_altitude.py ...
notebooks/
  Plotting_Examples.ipynb    # Colab-friendly gallery
docs/
  plotting_plan.md           # this file
```

This mirrors the physics layer: pure maths in `_coords.py` (like `grav_utils.py`), drawing code in
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

### Phase 0: Foundations (lead, about 1 week)

| Task | Description | Acceptance criteria |
|------|-------------|---------------------|
| 0.1 | Packaging: replace `packages = ["contigo"]` with `[tool.setuptools.packages.find]` and `include = ["contigo*"]`; add `plot = ["matplotlib>=3.7"]` extra; add `tests/` with a trivial passing test | `pip install .` (non-editable) in a fresh env gives `import contigo.forces` working; `pytest` runs |
| 0.2 | `contigo/contigo_utils/coords.py`: `ecef_to_geodetic` (vectorised), constants `WGS84_A`, `WGS84_F`, `WGS84_E2` into `constants.py` | Round-trip test vs forward formula < 1e-6 m; poles and equator tested; 1e6 points < 2 s |
| 0.3 | Shared SPICE kernel helper `check_pck()` in `time_utils.py` (same pattern as `check_lpsk`) that loads the PCK, checks coverage, downloads a newer one if needed, else warns and falls back to `IAU_EARTH`; `ecef_to_inertial(state_ecef, et)` using `sxform('ITRF93','J2000')` | Test: a point fixed on Earth's surface has inertial speed ~ 465 m/s at the equator; round trip ECEF to inertial and back < 1e-6 m; a date after 2026-05-30 either works (downloaded kernel) or warns and falls back |
| 0.5 | GitHub Actions workflow (lead): `ruff check` and `pytest` on every PR, Python 3.10 and 3.12 | PR shows green/red checks; the workflow installs `.[dev,plot]` only (no Java/GMAT), so tests must not need the JVM |
| 0.4 | `contigo/plotting/` skeleton: `_style.py`, `_util.py` (`as_spacecraft_dict`, `decimate`), empty `__init__.py`, CI-free smoke test using the `Agg` backend | `import contigo.plotting` works with and without matplotlib installed (clear error message when missing) |
| 0.5 | New SPIRE ECEF/GPS dataset | Load with pandas and readinto `Spacecraft` or `Constellation` |



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
def test_plot_altitude_runs(constellation_fixture):
    fig, ax = plot_altitude(constellation_fixture)
    assert len(ax.lines) == len(constellation_fixture)
```

| Task | Description | Acceptance criteria |
|------|-------------|---------------------|
| 1.1 | `plotting/altitude.py` with `plot_altitude` (alt vs time, one line per spacecraft, km) | Works on a single `Spacecraft` and a `Constellation`; returns `(fig, ax)`; legend lists IDs; x-axis is dates |
| 1.2 | Wrappers on `Spacecraft` and `Constellation` | Both call the same function; docstring links to it |
| 1.3 | Tests: pure-data test (known ECEF to expected altitude), smoke test (figure created, number of lines = number of IDs) | `pytest tests/plotting` green |
| 1.4 | Example cells in `notebooks/Plotting_Examples.ipynb` (with an "Open in Colab" badge) | Notebook runs top to bottom on the SPIRE cubesat dataset (**dependency:** Kyle supplies it; until then use `data/ESA_pod.hdf`) |

### Phase 2: Ground track and orbit plots

| Task | Description | Acceptance criteria |
|------|-------------|---------------------|
| 2.1 | `plot_ground_track`: lon/lat on a plain map with bundled coastline, lines **broken at the antimeridian** (insert NaN when `abs(dlon) > 180`), optional `color_by=` (time or altitude) | No horizontal streaks across the map; lat limits visible so inclination can be read; ground track of an ~87 deg inclination spacecraft (CHAMP) visibly reaches ±87. **Andreas had a really cool version of this plot in his last presentation**. |
| 2.2 | Wrappers + tests + notebook cells | As Phase 1 |
| 2.3 | `plot_orbit`: 3-D axes, Earth sphere (or oblate ellipsoid), orbit lines in J2000, `frame='inertial'` (default) or `'ecef'` | Equal-aspect 3-D axes; a circular orbit plots as a circle; `frame='ecef'` shows the spiral |
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

We work with **branches in the main repository**: you get write access, create a branch for
each task, and open a pull request (PR) into `main`. You cannot push directly to `main`; every
change is reviewed first (see 5.4). 

### 5.1 One-time setup

You need three things: a copy of the repository on your computer, a Python environment, and
a check that the tests pass. Pick **one** of the ways to get the repository (A or B). Option C
is for small edits only. Then do the Python setup, which is the same for everyone.

**Before you start:** install [Python 3.10 or newer](https://www.python.org/downloads/),
[Git](https://git-scm.com/downloads), and [VS Code](https://code.visualstudio.com/) with its
Python extension. Ask the lead to confirm you have write access to the repository (see 5.4).

#### A. Command line

```bash
git clone https://github.com/kylermurphy/contigo_edr.git
cd contigo_edr
```

Tell git who you are (once per computer):

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

#### B. GitHub Desktop (graphical, no commands)

1. Install [GitHub Desktop](https://desktop.github.com/) and sign in with your GitHub account
   (**File > Options > Accounts**, or **GitHub Desktop > Settings** on macOS).
2. Set your name and email under **Options > Git** (use the same email as your GitHub account).
3. Choose **File > Clone repository**, select the **URL** tab, enter
   `kylermurphy/contigo_edr`, pick a local folder you will remember, and click **Clone**.
4. To work on the code, click **Repository > Open in Visual Studio Code**
   (or **Open in Terminal** for the Python setup below).

#### C. GitHub in the browser (small edits only)

On any repository page, press the `.` key (or change `github.com` to `github.dev` in the address) to
open the web editor. You can read files and make small edits, such as to documentation, and
commit them to a branch. You **cannot** run Python, black, ruff, or pytest in it, so use A or B for
all code tasks. (GitHub Codespaces, a full cloud development environment, also works if the lead
enables it, but it is not required.)

#### Python environment (everyone, from the repository folder)

Open a terminal in the repository folder (in VS Code: **Terminal > New Terminal**).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e ".[dev,plot]"
pytest                             # should pass before you change anything
```

In VS Code, press **Ctrl+Shift+P** (Cmd+Shift+P on macOS), run **Python: Select Interpreter**, and
choose the `.venv` environment so the editor uses it.

If `pip install` fails (the full install includes Orekit and Java dependencies and can be slow or
fail on some systems), stop and ask, not everything has been tested and we should work through this; do not work around it silently.

#### GitHub Desktop equivalents for the steps in 5.2

If you use GitHub Desktop, use these buttons in place of the git commands in 5.2. The Python
checks (`black`, `ruff`, `pytest`) still run in the terminal.

| Step in 5.2 | Command line | GitHub Desktop |
|---|---|---|
| Update `main` | `git checkout main && git pull` | Set **Current Branch** to `main`, then click **Fetch origin** and **Pull origin** |
| Create a branch | `git checkout -b plot/altitude` | **Current Branch > New Branch**, name it, base it on `main` |
| Commit | `git add ... && git commit -m "..."` | Tick the changed files, write a summary, click **Commit to <branch>** |
| Push the branch | `git push -u origin plot/altitude` | Click **Publish branch** (later: **Push origin**) |
| Open a PR | follow the link printed by `git push` | Click **Create Pull Request** (opens GitHub in the browser) |
| After the merge | `git checkout main && git pull && git branch -d plot/altitude` | Switch to `main`, **Pull origin**, then **Branch > Delete** the old branch |
| See what changed | `git diff` | The **Changes** tab shows every edited file with a before/after view |

### 5.2 For every task

1. **Pick one task** from Section 4 and tell people  you are starting it.
2. **Start from an up-to-date `main`, then create a branch** named for the task:
   ```bash
   git checkout main
   git pull
   git checkout -b plot/altitude        # use a short, descriptive name
   ```
3. **Make small commits** with clear messages, for example `Add ecef_to_geodetic with tests`:
   ```bash
   git add path/to/changed_file.py
   git commit -m "Add ecef_to_geodetic with tests"
   ```
4. **Check before pushing:**
   ```bash
   black contigo tests          # formatting
   ruff check contigo tests     # lint
   pytest                       # tests
   ```
5. **Push your branch and open a pull request into `main`:**
   ```bash
   git push -u origin plot/altitude
   ```
   GitHub prints a link to open the PR. Fill in the checklist in Section 6. Automatic checks
   (ruff and pytest) run on every PR; a red cross means something needs fixing before review.
6. **Respond to review comments** by pushing more commits to the same branch. The PR updates
   automatically. Do not squash or force-push while a review is open.
7. **Review turnaround:** the lead reviews PRs within 24 hours. If you have heard nothing after
   that, send a reminder.
8. **After the PR is merged**, return to `main`, update it, and delete the old branch:
   ```bash
   git checkout main
   git pull
   git branch -d plot/altitude
   ```
   Start the next task from a fresh branch.

### 5.3 Working Rules

- One task per branch and one branch per PR. Do not stack unrelated changes together.
- If you are stuck I'm generally around questions. Early questions are cheap.
- Prefer a small PR (under ~300 changed lines) over a large one. Reviews are faster and bugs are easier to find.
- If `git pull` or a merge reports a conflict, stop and ask before resolving it.
- Never commit data files, notebook outputs with huge images, or credentials. Clear large outputs
  before committing (`jupyter nbconvert --clear-output`).
- A plot that "looks right" is not a test. Test the *numbers* going into the plot (Section 7).

### 5.4 Repository setup (lead, one time)

**

1. Add the contributor under **Settings > Collaborators** with **Write** access.
2. Protect `main` under **Settings > Branches > Add branch protection rule** (branch name `main`):
   - Require a pull request before merging, with **1 approval**.
   - Require status checks to pass (`ruff` and `pytest`) once CI (task 0.5) is in place.
   - Do not allow direct pushes to `main`.
3. In **Settings > General > Pull Requests**, enable **Automatically delete head branches** so merged
   branches are cleaned up.


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
- [ ] Example added to notebooks/Plotting_Examples.ipynb, run top to bottom
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

```python
"""Coordinate helpers for plotting.

added: DD/MM/YYYY Your Name <you@example.com>
"""
import numpy as np
import numpy.typing as npt

from contigo.contigo_utils.constants import WGS84_A, WGS84_E2


def ecef_to_geodetic(r_ecef: npt.ArrayLike, n_iter: int = 5) -> npt.NDArray[np.float64]:
    """Convert ECEF positions to WGS-84 geodetic coordinates.

    Parameters
    ----------
    r_ecef : array_like
        (..., 3) ECEF position [x, y, z] in metres.
    n_iter : int, optional
        Fixed-point iterations for latitude, by default 5.

    Returns
    -------
    numpy.ndarray
        (..., 3) [lat, lon, alt] with lat/lon in radians and alt in metres.
    """
    r = np.asarray(r_ecef, dtype=float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(n_iter):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + WGS84_E2 * n * np.sin(lat), p)
    # this form of the height is well behaved at the poles
    alt = (p * np.cos(lat) + z * np.sin(lat)
           - WGS84_A * np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2))
    return np.stack([lat, lon, alt], axis=-1)
```

(Constants: `WGS84_A = 6378137.0`, `WGS84_F = 1.0 / 298.257223563`,
`WGS84_E2 = WGS84_F * (2.0 - WGS84_F)`.) This implementation was checked against a forward
conversion for 100,000 random points (error < 1e-8 m) and at both poles.

### 7.2 Frame transform sketch (inertial)

```python
import spiceypy as spice

def ecef_to_inertial(state_ecef, et):
    """(N,6) ECEF state at SPICE ET times -> (N,6) J2000 state."""
    # check_pck() makes sure the leap-second and Earth PCK kernels are loaded
    out = np.empty_like(state_ecef)
    for i, t in enumerate(et):
        out[i] = spice.sxform("ITRF93", "J2000", t) @ state_ecef[i]
    return out
```

`sxform` returns the 6x6 state transform, so it includes Earth's rotation in the velocity
(`v_inertial = R (v_ecef + omega x r)`). It is a Python loop, so always decimate first.

### 7.3 A plot function skeleton

```python
def plot_altitude(data, *, ax=None, max_points=2000, ids=None, **plot_kwargs):
    import matplotlib.pyplot as plt                # lazy import
    from contigo.plotting import _util, _style
    from contigo.contigo_utils.coords import ecef_to_geodetic

    sc_dict = _util.as_spacecraft_dict(data, ids)  # {id: Spacecraft}
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    for sc_id, sc in sc_dict.items():
        idx = _util.decimate(sc.N, max_points)     # indices, evenly spaced
        alt_km = ecef_to_geodetic(sc.state_ecef[idx, :3])[:, 2] / 1e3
        ax.plot(sc.stime[idx], alt_km, color=_style.color_for(sc_id),
                label=str(sc_id), **plot_kwargs)

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Geodetic altitude (km)")
    ax.legend()
    return fig, ax
```

(Add a "no legend if more than ~15 spacecraft" guard so hundreds of lines don't produce a giant legend.)

### 7.4 Test templates

```python
# tests/plotting/test_coords.py  (numbers, no figures)
def test_equator_altitude():
    r = np.array([[WGS84_A + 400e3, 0.0, 0.0]])
    lat, lon, alt = ecef_to_geodetic(r)[0]
    assert alt == pytest.approx(400e3, abs=1e-3)

# tests/plotting/test_altitude.py  (smoke test)
import matplotlib
matplotlib.use("Agg")           # no display needed

def test_plot_altitude_one_line_per_spacecraft(constellation_fixture):
    fig, ax = plot_altitude(constellation_fixture)
    assert len(ax.lines) == len(constellation_fixture)
```

Tests should build a tiny synthetic `Spacecraft` from arrays (circular orbit, a few hundred points)
in a `conftest.py` fixture rather than load the large HDF files.

### 7.5 Gotchas to expect

- **Antimeridian:** a ground track jumps from +180 to -180; without breaking the line a horizontal
  streak crosses the map. Insert `np.nan` where `abs(np.diff(lon)) > np.pi`.
- **`sc_id` is a pandas category array**, so `np.unique` / dict keys work but be careful when comparing to strings.
- **Datetime axes:** `stime` is a pandas `DatetimeIndex`; matplotlib handles it directly.
- **3-D equal aspect:** `ax.set_box_aspect((1, 1, 1))`; a flat-looking Earth means this is missing.
- **Degrees vs radians:** transforms return radians. Convert once, at the plotting boundary.
- **Large `N`:** decimate with evenly spaced indices (`np.linspace(0, N - 1, max_points).astype(int)`)
  before transforms, not after.

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
| R1 | SPICE kernels must be loaded for the inertial transform | Task 0.3 shared helper; clear error if the PCK is missing |
| R2 | Plots with hundreds of spacecraft are visually unreadable | Default legend off above ~15 spacecraft; `ids=` subset; colormap by index |
| R3 | `Spacecraft` with several IDs and unsorted time | Always go through `split_by_id()` (sorts per ID) |
| R4 | Bundled Earth PCK ends 2026-05-30 (checked with `pckcov`); newer data fails to transform | Task 0.3: coverage check, download, `IAU_EARTH` fallback with warning; consider a periodic kernel refresh |
| R5 | Bundled coastline licence | Use Natural Earth (public domain), downsample to under ~100 KB |
| O1 | ~~Reviewer / turnaround~~ | **Resolved:** Kyle, within 24 hours |
| O2 | ~~CI~~ | **Resolved:** ruff + pytest, task 0.5, done by Kyle |
| O3 | ~~Example dataset~~ | **Resolved:** new SPIRE cubesat dataset (Kyle to create); synthetic fixtures for tests |
| O4 | ~~Docs site~~ | **Resolved:** later |
| O5 | **[OPEN]** Which contributor takes Phase 1 and when does Phase 0 land? | Kevin takes Phase 1, Kyle working on Phase 0 now |
