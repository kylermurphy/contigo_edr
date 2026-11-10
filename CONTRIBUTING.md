# Contributing to CONTIGO

Thanks for helping. This guide covers setting up, the workflow for every change, and the rules
we follow. If anything here is unclear or does not work, ask: early questions are cheap.

We work with **branches in the main repository**. You get write access, create a branch for
each task, and open a pull request (PR) into `main`. Nobody pushes directly to `main`; every
change is reviewed first, and the lead reviews PRs within 24 hours.

## 1. One-time setup

You need a copy of the repository, a Python environment, and a check that the tests pass.

**Before you start:** install [Python 3.10 or newer](https://www.python.org/downloads/),
[Git](https://git-scm.com/downloads), and [VS Code](https://code.visualstudio.com/) with its
Python extension. Ask the lead to confirm you have write access.

### Get the repository (pick one)

**A. Command line**

```bash
git clone https://github.com/kylermurphy/contigo_edr.git
cd contigo_edr
git config --global user.name "Your Name"         # once per computer
git config --global user.email "you@example.com"
```

**B. GitHub Desktop (graphical)**

1. Install [GitHub Desktop](https://desktop.github.com/) and sign in (**File > Options >
   Accounts**, or **GitHub Desktop > Settings** on macOS).
2. Set your name and email under **Options > Git** (same email as your GitHub account).
3. **File > Clone repository**, **URL** tab, enter `kylermurphy/contigo_edr`, pick a folder,
   **Clone**.
4. **Repository > Open in Visual Studio Code** (or **Open in Terminal**) for the Python setup.

**C. GitHub in the browser (small edits only).** Press `.` on any repository page to open the
web editor. Good for documentation fixes; you cannot run Python or the tests there.

### Python environment (everyone, from the repository folder)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e ".[dev,plot]"
pytest                             # should pass before you change anything
```

In VS Code, run **Python: Select Interpreter** (Ctrl+Shift+P) and choose `.venv`.

If `pip install` fails (Orekit and its bundled Java can be slow or fail on some systems), stop
and ask rather than working around it. See [docs/installation.md](docs/installation.md) for
what gets installed and downloaded.

## 2. For every change

1. **Pick one task** and tell people you are starting it.
2. **Start from an up-to-date `main` and create a branch** with a short, descriptive name:
   ```bash
   git checkout main
   git pull
   git checkout -b plot/altitude
   ```
3. **Make small commits** with clear messages:
   ```bash
   git add path/to/changed_file.py
   git commit -m "Add plot_altitude with tests"
   ```
4. **Check before pushing:**
   ```bash
   black contigo tests          # formatting
   ruff check contigo tests     # lint
   pytest                       # tests
   ```
5. **Push the branch and open a PR into `main`:**
   ```bash
   git push -u origin plot/altitude
   ```
   GitHub prints a link to open the PR. Fill in the checklist (section 5). CI runs lint and
   the tests on every PR; a red cross means something needs fixing before review.
6. **Respond to review comments** by pushing more commits to the same branch. Do not squash or
   force-push while a review is open.
7. **No reply after 24 hours?** Send a reminder.
8. **After the merge**, update `main` and delete your branch, then start the next task from a
   fresh branch:
   ```bash
   git checkout main
   git pull
   git branch -d plot/altitude
   ```

### GitHub Desktop equivalents

| Step | Command line | GitHub Desktop |
|------|--------------|----------------|
| Update `main` | `git checkout main && git pull` | **Current Branch** `main`, then **Fetch origin** and **Pull origin** |
| Create a branch | `git checkout -b plot/altitude` | **Current Branch > New Branch**, based on `main` |
| Commit | `git add ... && git commit -m "..."` | Tick the files, write a summary, **Commit to <branch>** |
| Push | `git push -u origin plot/altitude` | **Publish branch** (later **Push origin**) |
| Open a PR | follow the link printed by `git push` | **Create Pull Request** |
| After the merge | `git checkout main && git pull && git branch -d plot/altitude` | Switch to `main`, **Pull origin**, **Branch > Delete** |
| See what changed | `git diff` | the **Changes** tab |

The Python checks (`black`, `ruff`, `pytest`) still run in a terminal.

## 3. Rules

- One task per branch, one branch per PR. Do not mix unrelated changes.
- Prefer small PRs (under ~300 changed lines): faster reviews, easier bug spotting.
- If `git pull` or a merge reports a conflict, stop and ask before resolving it.
- **Never commit** data files, credentials, or notebook outputs with large images.
- **Notebooks:** clear outputs before committing (`jupyter nbconvert --clear-output --inplace
  notebook.ipynb`). Each notebook in `notebooks/tutorials/` must stay under 1 MB; a test
  enforces it. Tutorials use example data dated **before 2026-05-30** so they work offline
  (see [docs/coordinates.md](docs/coordinates.md)).
- New behaviour needs tests that check the **numbers**, not only that code runs. See
  [docs/testing.md](docs/testing.md).
- Add a line to [CHANGELOG.md](CHANGELOG.md) under "Unreleased" for anything a user would
  notice.

## 4. Code style

Follow the existing code (`spacecraft.py`, `forces/base.py`, `contigo_utils/coords.py`).

- Module docstring with an `added: DD/MM/YYYY Name <email>` line.
- Absolute imports (`from contigo.contigo_utils import kernels`), and
  `logger = logging.getLogger(__name__)`. Never `print`, and never configure logging in the
  package.
- numpy-style docstrings (Parameters / Returns / Raises) and type hints on public functions
  (`numpy.typing` for arrays).
- Line length 88 (black), `snake_case`, private helpers start with `_`.
- Internal units are SI (m, m/s) and frames are ECEF; angles in radians. Convert at the edges.
- Raise `ValueError` with a message that names the bad value or shape.
- Optional dependencies (matplotlib) are imported inside the functions that need them; nothing
  starts the JVM or uses the network at import.
- New tests that need the network, Java or more than ~10 s get the `network`, `orekit` or
  `slow` marker.

## 5. Pull request checklist

Copy this into the PR description:

```
- [ ] What and why in one or two sentences (and the task ID, if any)
- [ ] black, ruff and pytest pass locally
- [ ] Tests added or updated; they check numbers, with the source of each expected value
- [ ] Public functions have numpy-style docstrings and type hints
- [ ] CHANGELOG.md updated (if a user would notice)
- [ ] Docs updated (README, docs/) if behaviour or usage changed
- [ ] Notebook outputs cleared; tutorials under 1 MB
- [ ] Screenshot attached (plot changes only)
```

Plotting tasks have extra items; see [docs/plotting_plan.md](docs/plotting_plan.md) section 6.

## 6. Maintainers: repository settings

- **Settings > Collaborators:** contributors get **Write** access.
- **Settings > Branches:** `main` requires a PR with 1 approval and passing checks (`lint` and
  `test (3.10)`, `test (3.12)`, `test (3.13)`, `test (3.14)`); no direct pushes (the owner can
  bypass).
- **Settings > General > Pull Requests:** enable **Automatically delete head branches**.
