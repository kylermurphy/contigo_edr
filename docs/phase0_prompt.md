# Phase 0 starter prompt for Claude Code (VS Code)

Paste everything inside the block below into a new Claude Code session opened at the
`contigo_edr` repo root. Tip: start the session in **Plan mode** (Shift+Tab twice) so Claude
proposes each task before editing, and use the VS Code Source Control panel to review every diff.

Before you start, make sure the plan is on the branch you work from (merge the plan PR into
`main`, or check out `claude/contigo-plotting-module-plan-9o6mrd`).

````text
You are helping me implement Phase 0 of the CONTIGO plotting module. Work step by step and
stop for my review after each task. I want to understand what changes and why.

## 1. Read first (do not edit anything yet)
1. `docs/plotting_plan.md`: the full work plan. Phase 0 is Section 4 (tasks 0.1 to 0.5);
   the code templates are in Section 7; decisions D1 to D8 and risks R1 to R5 are binding.
2. Existing code to match in style: `contigo/spacecraft.py`, `contigo/constellation.py`,
   `contigo/config.py`, `contigo/contigo_utils/time_utils.py`, `contigo/contigo_utils/constants.py`,
   `contigo/contigo_utils/jb08_utils.py` (for validation and error-message style),
   `contigo/ephemeris/spice_ephem.py` (especially `load_kernels` and `dl_kernels`, which
   the PCK helper should reuse the logic of), and `pyproject.toml`.
3. Summarise back to me in under 15 lines: the style conventions you will follow, how kernels
   are loaded today, and anything in the plan that looks wrong or unclear given the code.
   Then wait for my go-ahead.

## 2. Ground rules
- One branch and one commit series per task, named `phase0/<short-task-name>`, branched from the
  branch I am on now. Do NOT push, open a PR, or touch `main` unless I ask.
- Small commits with clear messages. Do not rewrite history.
- Follow the existing style: module docstring with an `added: DD/MM/YYYY Kyle Murphy <kylemurphy.spacephys@gmail.com>`
  line, numpy-style docstrings, type hints with `numpy.typing`, absolute `contigo.` imports,
  `logging.getLogger(__name__)`, line length 88.
- Keep changes minimal and in scope. Do not refactor unrelated code, and tell me about
  anything unrelated you notice instead of fixing it.
- Tests must not need Java/Orekit, GMAT, network access, or the large data files. Use small
  synthetic arrays. Use the `Agg` matplotlib backend for any figure tests.
- Install only what is needed to run the tests; the full `pip install -e ".[dev]"` pulls in
  heavy Orekit/JVM dependencies. If a dependency is missing, tell me what you want to install
  and why before installing it.
- Before declaring a task done, run and show me the output of: `black --check`, `ruff check`,
  and `pytest`, restricted to the files you touched when the rest of the repo has existing
  failures (report those separately, do not fix them).
- After each task, give me: files changed, what each change does in plain language, how to
  run the checks, and any decision you made that I should confirm.

## 3. Tasks (in this order, one at a time)

**0.1 Packaging and test scaffold**
- In `pyproject.toml`, replace `packages = ["contigo"]` with
  `[tool.setuptools.packages.find]` and `include = ["contigo*"]`; make sure the existing
  `package-data` still points at real paths (the current `contigo.orekit_utils` entry looks
  wrong; check and tell me before changing it).
- Add the optional extra `plot = ["matplotlib>=3.7"]`.
- Check `[tool.ruff]`: newer ruff expects `select`/`ignore` under `[tool.ruff.lint]`. Fix only
  if ruff warns, and tell me.
- Create `tests/conftest.py` and `tests/test_smoke.py` (no `__init__.py` needed)
  with one trivial passing test.
- Verify with a non-editable install into a throwaway virtualenv (in the scratchpad or
  /tmp, not the repo) that `import contigo.forces` and `import contigo.ephemeris` work.
  If the install is too heavy because of dependencies, instead verify with
  `python -m build` or `pip install --no-deps` and `pip show -f contigo` that subpackages are included.

**0.2 Geodetic conversion**
- Add `WGS84_A`, `WGS84_F`, `WGS84_E2` to `contigo/contigo_utils/constants.py`.
- Create `contigo/contigo_utils/coords.py` with `ecef_to_geodetic` exactly as designed in
  Section 7.1 of the plan (vectorised, `(..., 3)` in metres to `(..., 3)` of lat, lon in
  radians and altitude in metres). Add input validation (last axis must be 3, finite values).
- Add `tests/test_coords.py`: equator, both poles, a round trip against the forward formula
  on 100k random points (error < 1e-6 m), shape handling for `(3,)`, `(N,3)`, `(M,N,3)`,
  and a speed sanity check on 1e6 points that only asserts a generous bound.

**0.3 PCK helper and inertial transform**
- In `contigo/contigo_utils/time_utils.py` add `check_pck()` following the `check_lpsk` pattern:
  load `config.PCK_FILE`, check coverage with `spice.pckcov(file, 3000)` against the epochs
  requested, download a newer `earth_latest_high_prec.bpc` when coverage is insufficient
  (reuse the URL logic from `SPICEEphem.dl_kernels`, but do not download during tests),
  and otherwise warn via `logger.warning` and fall back to the `IAU_EARTH` frame. The bundled
  kernel ends 2026-05-30; this is risk R4 in the plan.
- Add `ecef_to_inertial(state_ecef, et)` to `coords.py` using `spice.sxform('ITRF93','J2000', t)`
  per epoch (Section 7.2). Return `(N,6)`. Document that callers should decimate first.
  Also add the inverse `inertial_to_ecef`.
- Tests (`tests/test_coords.py` or `tests/test_frames.py`): a point fixed on the equator has
  inertial speed ~465.1 m/s; round trip error < 1e-6 m; a date after the kernel coverage
  triggers the warning/fallback path (mock the download so no network is used).
  SPICE kernels in `contigo/data/` (`naif0012.tls`, `earth_latest_high_prec.bpc`) are
  available for tests.

**0.4 Plotting package skeleton**
- Create `contigo/plotting/` with `__init__.py`, `_style.py` (`color_for(sc_id)` giving a stable
  colour per ID from a colour-blind-safe palette), and `_util.py` (`as_spacecraft_dict(data, ids=None)`
  accepting a `Spacecraft` or `Constellation`, and `decimate(n, max_points)` returning evenly
  spaced indices). Import matplotlib lazily; `import contigo.plotting` must work without
  matplotlib installed, and plot functions must raise a clear ImportError mentioning
  `pip install contigo[plot]`.
- Tests with synthetic `Spacecraft` objects built from arrays (a circular orbit, a few hundred
  points, two IDs). Put the fixture in `tests/conftest.py`.
- Do not implement any plots yet (that is Phase 1).

**0.5 CI (ruff and pytest)**
- Add `.github/workflows/ci.yml`: on pull_request and push to main, Python 3.10 and 3.12,
  `pip install -e ".[dev,plot]"` is too heavy, so install only what the tests need
  (numpy, pandas, scipy, spiceypy, matplotlib, pytest, ruff) or add a lean `test` extra and
  explain your choice to me. Run `ruff check contigo tests` and `pytest`.
- If `ruff check` fails on existing code, tell me and propose either fixing, or scoping CI to
  new files for now. Do not mass-reformat the repo.

## 4. Start
Do step 1 (read and summarise) now, then wait for my go-ahead before touching any file.
````

## Notes for the lead

- Review each task's diff in the Source Control panel before telling Claude to continue.
- If tests need `pyshtools`/`numba` because importing `contigo.spacecraft` pulls other modules, that is
  useful information. Ask Claude to report it rather than work around it.
- Tasks are independent enough that 0.1 and 0.2 can be reviewed and merged first.
