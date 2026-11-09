"""Package-level checks: import side effects and tutorial notebook imports.

PK-01  importing contigo modules must not start the JVM (fresh subprocess).
PK-01b/c  static check of the two modules that still start it at import.
NB-01  every ``from contigo... import X`` in the tutorial notebooks resolves,
       checked by parsing sources only (no contigo module is executed).

Expected values: ``docs/phase0_plan.md`` sections 2 (F4, F9), 4.7 and R3.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest

import contigo

REPO = Path(__file__).resolve().parents[1]

#: Modules that start the JVM when imported today. They are never imported
#: here (that would start Java and contact GitLab); they are checked with ast.
JVM_AT_IMPORT = {
    "contigo.forces.srp_orekit": "F4/R3",
    "contigo.contigo_utils.jb08_utils": "same pattern as F4/R3",
}

PRUNE_DIRS = {
    ".git",
    ".claude",
    ".ipynb_checkpoints",
    ".venv",
    "node_modules",
    "java_src",
    "__pycache__",
    "build",
    "dist",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _all_modules() -> list[str]:
    return sorted(
        m.name for m in pkgutil.walk_packages(contigo.__path__, prefix="contigo.")
    )


def _module_source_path(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, f"cannot locate {name}"
    return Path(spec.origin)


class _ImportTimeJVM(ast.NodeVisitor):
    """Collect JVM starts and Java imports executed when the module is imported.

    Walks module-level code (including ``if``/``try`` blocks) but never
    descends into function or class bodies, which run later.
    """

    def __init__(self) -> None:
        self.hits: list[str] = []

    def visit_FunctionDef(self, node):  # noqa: N802
        pass

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815
    visit_ClassDef = visit_FunctionDef  # noqa: N815
    visit_Lambda = visit_FunctionDef  # noqa: N815

    def visit_Call(self, node):  # noqa: N802
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name == "start_orekit":
            self.hits.append(f"line {node.lineno}: call to start_orekit()")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):  # noqa: N802
        mod = node.module or ""
        if mod.split(".")[0] in {"org", "java"}:
            self.hits.append(f"line {node.lineno}: from {mod} import ...")

    def visit_Import(self, node):  # noqa: N802
        for alias in node.names:
            if alias.name.split(".")[0] in {"org", "java"}:
                self.hits.append(f"line {node.lineno}: import {alias.name}")


def import_time_jvm_use(source: str) -> list[str]:
    """Return the module-level JVM starts and Java imports found in ``source``."""
    visitor = _ImportTimeJVM()
    visitor.visit(ast.parse(source))
    return visitor.hits


_IMPORT_PROBE = r"""
import importlib, json, sys
import jpype

started = []
real_start = jpype.startJVM
def guard(*a, **k):
    raise RuntimeError("JVM start attempted")
jpype.startJVM = guard

failed = {}
for name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(name)
    except BaseException as exc:  # report, do not stop
        failed[name] = repr(exc)
    if jpype.isJVMStarted():
        started.append(name)
        break
print("RESULT:" + json.dumps({"file": importlib.import_module("contigo").__file__,
                              "failed": failed, "started": started,
                              "jvm": jpype.isJVMStarted()}))
"""


# ---------------------------------------------------------------------------
# PK-01
# ---------------------------------------------------------------------------
def test_pk01_import_does_not_start_jvm(tmp_path):
    """PK-01: importing every submodule leaves the JVM stopped.

    Source: plan 4.7 (``jpype.isJVMStarted()`` stays False). Runs in a fresh
    interpreter with ``tmp_path`` as cwd, and ``PYTHONPATH`` set to the copy of
    ``contigo`` under test so the same code is imported. ``jpype.startJVM`` is
    replaced by a raising guard, so a regression cannot hang on Java or GitLab.
    The modules in ``JVM_AT_IMPORT`` are covered by PK-01b/PK-01c instead.
    """
    names = [n for n in _all_modules() if n not in JVM_AT_IMPORT]
    assert "contigo.edr_efd" in names and "contigo.forces.srp_gmat" in names

    pkg_root = str(Path(contigo.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [pkg_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, json.dumps(names)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert lines, f"probe produced no result:\n{proc.stdout}\n{proc.stderr}"
    result = json.loads(lines[-1][len("RESULT:") :])

    assert Path(result["file"]).resolve().parent.parent == Path(pkg_root).resolve()
    assert result["started"] == [], f"JVM started by {result['started']}"
    assert result["jvm"] is False
    assert result["failed"] == {}, f"modules failed to import: {result['failed']}"


def test_pk01_scanner_distinguishes_import_time_from_deferred():
    """PK-01 helper: the ast scan flags module-level use only (self-check).

    Expected: top-level and ``if``-guarded calls/imports are found; the same
    statements inside a function or method are not.
    """
    flagged = import_time_jvm_use(
        "import contigo.config as c\n"
        "if c.state['orekit_loaded'] is False:\n"
        "    orekit_utils.start_orekit()\n"
        "from org.orekit.time import AbsoluteDate\n"
        "import java.util\n"
    )
    assert len(flagged) == 3
    deferred = import_time_jvm_use(
        "class A:\n"
        "    def __init__(self):\n"
        "        from contigo.contigo_utils import orekit_utils\n"
        "        orekit_utils.start_orekit()\n"
        "        from org.orekit.time import AbsoluteDate\n"
        "def f():\n"
        "    import org.orekit\n"
    )
    assert deferred == []


def test_pk01_jvm_modules_are_still_the_known_ones():
    """PK-01 guard: only the modules in ``JVM_AT_IMPORT`` use the JVM at import.

    A new module that starts the JVM at import must be fixed, not added to the
    list; the list may only shrink.
    """
    offenders = {
        name
        for name in _all_modules()
        if import_time_jvm_use(_module_source_path(name).read_text(encoding="utf-8"))
    }
    assert offenders <= set(
        JVM_AT_IMPORT
    ), f"new import-time JVM use: {sorted(offenders - set(JVM_AT_IMPORT))}"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F4/R3: srp_orekit starts the JVM at import",
)
def test_pk01b_srp_orekit_does_not_start_jvm_at_import():
    """PK-01b: ``forces.srp_orekit`` must not start the JVM when imported.

    Checked statically (importing it would start Java and query GitLab).
    Expected after R3: no module-level ``start_orekit`` call or ``org.*``
    import. Today both are present (lines 15-20).
    """
    path = _module_source_path("contigo.forces.srp_orekit")
    assert import_time_jvm_use(path.read_text(encoding="utf-8")) == []


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="same pattern as F4/R3: jb08_utils starts the JVM at import",
)
def test_pk01c_jb08_utils_does_not_start_jvm_at_import():
    """PK-01c: ``contigo_utils.jb08_utils`` must not start the JVM when imported.

    New finding in the PK-01 sweep: it repeats the ``srp_orekit`` pattern
    (module-level ``start_orekit`` and ``org.*`` imports). Static check, as
    PK-01b.
    """
    path = _module_source_path("contigo.contigo_utils.jb08_utils")
    assert import_time_jvm_use(path.read_text(encoding="utf-8")) == []


# ---------------------------------------------------------------------------
# NB-01
# ---------------------------------------------------------------------------
def _find_notebook(name: str) -> Path | None:
    """Locate ``name`` anywhere in the repo (survives the move to notebooks/tutorials)."""
    found = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in PRUNE_DIRS]
        if name in files:
            found.append(Path(root) / name)
    found.sort(key=lambda p: ("tutorials" not in p.parts, len(p.parts)))
    return found[0] if found else None


_IMPORT_LINE = re.compile(
    r"^\s*(from\s+contigo[\w.]*\s+import\s+[\w, *]+|import\s+contigo)"
)


def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        cells.append("".join(src) if isinstance(src, list) else src)
    return cells


def _neutralise_magics(cell: str) -> str | None:
    """Replace ``!``/``%`` lines by ``pass``; ``None`` for a ``%%`` cell magic."""
    lines = cell.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        return None
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            out.append(line[: len(line) - len(stripped)] + "pass")
        else:
            out.append(line)
    return "\n".join(out)


def contigo_imports(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(module, names)`` for each contigo import in the code cells.

    ``import contigo.X`` gives ``(contigo.X, ())``.
    """
    found: list[tuple[str, tuple[str, ...]]] = []
    for cell in _code_cells(path):
        text = _neutralise_magics(cell)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # IPython syntax such as ``x = !cmd``: keep only the import lines
            kept = [ln.strip() for ln in text.splitlines() if _IMPORT_LINE.match(ln)]
            tree = ast.parse("\n".join(kept))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                mod = node.module or ""
                if mod == "contigo" or mod.startswith("contigo."):
                    found.append((mod, tuple(a.name for a in node.names)))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "contigo" or alias.name.startswith("contigo."):
                        found.append((alias.name, ()))
    return found


def _bound_names(body: list[ast.stmt]) -> set[str]:
    """Names bound at module level, descending into ``if``/``try``/``with``."""
    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.If, ast.With, ast.For, ast.While)):
            names |= _bound_names(node.body + node.orelse)
        elif isinstance(node, ast.Try):
            handlers = [s for h in node.handlers for s in h.body]
            names |= _bound_names(node.body + handlers + node.orelse + node.finalbody)
    return names


def _find_spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None


def unresolved_imports(path: Path) -> list[str]:
    """Describe every contigo import of the notebook that does not resolve."""
    problems: list[str] = []
    for module, names in contigo_imports(path):
        spec = _find_spec(module)
        if spec is None:
            problems.append(f"{module}: module not found")
            continue
        if not names or names == ("*",):
            continue
        defined: set[str] = set()
        if spec.origin and spec.origin.endswith(".py"):
            tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
            defined = _bound_names(tree.body)
        for name in names:
            if name in defined or _find_spec(f"{module}.{name}") is not None:
                continue
            problems.append(f"{module}.{name}: name not defined")
    return problems


def test_nb01_helpers_find_missing_names(tmp_path):
    """NB-01 helper: a bad name and a bad module are reported, good ones pass.

    Expected: ``Constellation`` and the submodule ``spacecraft`` resolve;
    ``ThirdBody`` (removed) and a nonexistent module do not.
    """
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "!pip install contigo\n",
                    "%matplotlib inline\n",
                    "from contigo.constellation import Constellation\n",
                    "from contigo import spacecraft\n",
                    "from contigo.forces.third_body_acc import ThirdBody, ThirdBodyEnv\n",
                    "import contigo.nope\n",
                ],
            },
            {"cell_type": "markdown", "source": ["from contigo.fake import Nothing"]},
        ]
    }
    path = tmp_path / "x.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    assert unresolved_imports(path) == [
        "contigo.forces.third_body_acc.ThirdBody: name not defined",
        "contigo.nope: module not found",
    ]


@pytest.mark.parametrize(
    "notebook",
    [
        pytest.param(
            "CONTIGO_Colab.ipynb",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="F9/N2: CONTIGO_Colab imports removed names",
            ),
        ),
        "Orekit_EDR.ipynb",
        "Tutorial_Overview.ipynb",
    ],
)
def test_nb01_notebook_imports_resolve(notebook):
    """NB-01: every contigo import in the tutorial notebook resolves.

    Static check only (ast on the notebook cells and on the contigo sources).
    Expected: no unresolved module or name. ``CONTIGO_Colab`` imports
    ``ThirdBody`` from ``third_body_acc``, which no longer exists (F9/N2).
    """
    path = _find_notebook(notebook)
    if path is None:
        pytest.skip(f"{notebook} not found in the repository (sdist?)")
    imports = contigo_imports(path)
    if not imports:
        pytest.skip(f"{notebook} has no contigo imports to check")
    assert unresolved_imports(path) == []
