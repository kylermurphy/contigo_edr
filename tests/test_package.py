"""Package-level checks: import side effects and tutorial notebook imports.

PK-01  importing contigo modules must not start the JVM (fresh subprocess).
PK-01b/c  static check of the two modules that still start it at import.
NB-01  every ``from contigo... import X`` in the tutorial notebooks resolves,
       checked by parsing sources only: no contigo module is executed (only
       the lightweight package ``__init__`` files are, through ``find_spec``).

Expected values: ``docs/phase0_plan.md`` sections 2 (F4, F9), 4.7 and R3.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

import contigo

REPO = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

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

#: Calls that start or configure the JVM (matched by function or attribute name).
JVM_CALLS = {
    "start_orekit",
    "initVM",
    "startJVM",
    "setup_orekit_curdir",
    "setup_orekit_data",
    "download_orekit_data_curdir",
    "JClass",
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


def _probe_env() -> dict[str, str]:
    """Environment that makes a subprocess import the ``contigo`` under test."""
    pkg_root = str(Path(contigo.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [pkg_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return env


def _run_probe(script: str, args: list[str], cwd: Path) -> dict:
    """Run ``script`` in a fresh interpreter; return its ``RESULT:`` JSON."""
    proc = subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=cwd,
        env=_probe_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert lines, f"probe produced no result:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(lines[-1][len("RESULT:") :])


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
        if name in JVM_CALLS:
            self.hits.append(f"line {node.lineno}: call to {name}()")
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
import importlib, json, sys, traceback
import jpype

attempts = []
current = [None]
def guard(*a, **k):
    tb = traceback.format_stack(limit=6)[:-1]
    attempts.append({"module": current[0], "traceback": "".join(tb)[-600:]})
    raise RuntimeError("JVM start attempted")
jpype.startJVM = guard

failed = {}
for name in json.loads(sys.argv[1]):
    current[0] = name
    try:
        importlib.import_module(name)
    except Exception as exc:  # report, do not stop
        failed[name] = repr(exc)
print("RESULT:" + json.dumps({"file": importlib.import_module("contigo").__file__,
                              "failed": failed, "attempts": attempts,
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
    replaced by a recording guard that raises, so a regression is reported by
    module and cannot hang on Java or GitLab. The modules in ``JVM_AT_IMPORT``
    are covered by PK-01b/PK-01c instead.
    """
    names = [n for n in _all_modules() if n not in JVM_AT_IMPORT]
    assert "contigo.edr_efd" in names and "contigo.forces.srp_gmat" in names

    result = _run_probe(_IMPORT_PROBE, [json.dumps(names)], tmp_path)

    pkg_root = Path(contigo.__file__).resolve().parent.parent
    assert Path(result["file"]).resolve().parent.parent == pkg_root
    started = [
        f"JVM start attempted by {a['module']}:\n{a['traceback']}"
        for a in result["attempts"]
    ]
    assert started == []
    assert result["jvm"] is False
    assert result["failed"] == {}, f"modules failed to import: {result['failed']}"


@pytest.mark.parametrize("call", sorted(JVM_CALLS))
@pytest.mark.parametrize("form", ["{c}()", "mod.{c}(1, x=2)"])
def test_pk01_scanner_flags_each_jvm_call(call, form):
    """PK-01 helper: every call in ``JVM_CALLS`` is flagged at module level.

    Expected: one hit for the bare and the attribute form, inside ``if`` too;
    none when the same call is inside a function.
    """
    stmt = form.format(c=call)
    assert len(import_time_jvm_use(stmt + "\n")) == 1
    assert len(import_time_jvm_use(f"if x:\n    {stmt}\n")) == 1
    assert import_time_jvm_use(f"def f():\n    {stmt}\n") == []


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
# NB-01: locating and parsing notebooks
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _notebook_index() -> dict[str, Path]:
    """Map notebook file name to path, walking the repo once.

    When a name occurs twice the copy under a ``tutorials`` folder wins, then
    the shallowest. Surviving the later move to ``notebooks/tutorials``.
    """
    found: dict[str, list[Path]] = {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in PRUNE_DIRS]
        for f in files:
            if f.endswith(".ipynb"):
                found.setdefault(f, []).append(Path(root) / f)
    return {
        name: min(paths, key=lambda p: ("tutorials" not in p.parts, len(p.parts)))
        for name, paths in found.items()
    }


def _find_notebook(name: str) -> Path | None:
    return _notebook_index().get(name)


def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        cells.append("".join(src) if isinstance(src, list) else src)
    return cells


#: ``x = !cmd``, ``a, b = %magic`` and ``x = ?obj`` (IPython assignment forms)
_IPY_ASSIGN = re.compile(r"^[\w.,\s\[\]()]+?=\s*[!%]")


def _neutralise_ipython(cell: str) -> str | None:
    """Replace IPython-only lines by ``pass`` at the same indentation.

    Lines starting with ``!`` or ``%`` and ``x = !cmd`` forms (with their
    backslash-continued lines) become ``pass``. Returns ``None`` for a
    ``%%`` cell magic, whose body is not Python.
    """
    lines = cell.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        return None
    out: list[str] = []
    continued = False
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if continued:
            continued = line.rstrip().endswith("\\")
            continue
        if stripped.startswith(("!", "%")) or _IPY_ASSIGN.match(stripped):
            out.append(indent + "pass")
            continued = line.rstrip().endswith("\\")
        else:
            out.append(line)
    return "\n".join(out)


def contigo_imports(
    path: Path, notebook: str | None = None
) -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(module, names)`` for each contigo import in the code cells.

    ``import contigo.X`` gives ``(contigo.X, ())``. Fails the test with the
    notebook and cell number if a cell cannot be parsed after IPython syntax
    is replaced.
    """
    found: list[tuple[str, tuple[str, ...]]] = []
    for i, cell in enumerate(_code_cells(path)):
        text = _neutralise_ipython(cell)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            pytest.fail(
                f"{notebook or path.name}: code cell {i} does not parse "
                f"({exc.msg}, line {exc.lineno}):\n{cell[:300]}"
            )
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


# ---------------------------------------------------------------------------
# NB-01: name resolution without executing modules
# ---------------------------------------------------------------------------
StarResolver = Callable[[ast.ImportFrom], "set[str]"]


def _store_names(node: ast.AST) -> set[str]:
    """Names assigned (``Store`` context) anywhere inside ``node``."""
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }


def _walk_scope(node: ast.AST):
    """Yield ``node`` and its descendants, skipping nested function/class bodies."""
    yield node
    if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    ):
        return
    for child in ast.iter_child_nodes(node):
        yield from _walk_scope(child)


def _bound_names(body: list[ast.stmt], star: StarResolver | None = None) -> set[str]:
    """Names bound at module level of ``body``.

    Handles def/class, (annotated, augmented) assignment, imports, ``with``
    targets, ``for`` targets, ``except ... as``, ``match`` captures, walrus
    targets, ``type`` aliases, and ``if``/``while``/``try``/``with``/``for``
    bodies. A ``from x import *`` adds ``star(node)`` when a resolver is given.
    """
    names: set[str] = set()
    try_types = tuple(
        t for t in (ast.Try, getattr(ast, "TryStar", None)) if t is not None
    )
    for stmt in body:
        for sub in _walk_scope(stmt):
            if isinstance(sub, ast.NamedExpr):
                names.add(sub.target.id)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                names |= _store_names(target)
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
            names |= _store_names(stmt.target)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.name == "*":
                    if star is not None:
                        names |= star(stmt)
                else:
                    names.add(alias.asname or alias.name)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None:
                    names |= _store_names(item.optional_vars)
            names |= _bound_names(stmt.body, star)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            names |= _store_names(stmt.target)
            names |= _bound_names(stmt.body + stmt.orelse, star)
        elif isinstance(stmt, (ast.If, ast.While)):
            names |= _bound_names(stmt.body + stmt.orelse, star)
        elif isinstance(stmt, try_types):
            for handler in stmt.handlers:
                if handler.name:
                    names.add(handler.name)
                names |= _bound_names(handler.body, star)
            names |= _bound_names(stmt.body + stmt.orelse + stmt.finalbody, star)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                for pat in ast.walk(case.pattern):
                    for attr in ("name", "rest"):
                        val = getattr(pat, attr, None)
                        if isinstance(val, str) and not isinstance(pat, ast.MatchClass):
                            names.add(val)
                names |= _bound_names(case.body, star)
        elif type(stmt).__name__ == "TypeAlias":  # ``type X = ...`` (3.12)
            names.add(stmt.name.id)
    return names


def _find_spec(name: str):
    """``find_spec`` that imports only the parent packages, never ``name``."""
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None


def _is_contigo(module: str) -> bool:
    return module == "contigo" or module.startswith("contigo.")


def _star_names(node: ast.ImportFrom, package: str) -> set[str]:
    """Names a ``from x import *`` brings in: those bound in ``x``, one level."""
    try:
        mod = importlib.util.resolve_name(
            "." * node.level + (node.module or ""), package
        )
    except ImportError:
        return set()
    if not _is_contigo(mod):
        return set()
    spec = _find_spec(mod)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return set()
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    return _bound_names(tree.body)


def _name_resolves(module: str, name: str, spec) -> bool:
    """Is ``name`` a submodule of, or bound in, ``module`` (no module executed)?"""
    if spec.submodule_search_locations is not None:  # a package
        for loc in spec.submodule_search_locations:
            base = Path(loc)
            if (base / f"{name}.py").is_file() or (
                base / name / "__init__.py"
            ).is_file():
                return True
    if not spec.origin or not spec.origin.endswith(".py"):
        return False
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    package = spec.parent or module
    defined = _bound_names(tree.body, lambda node: _star_names(node, package))
    return name in defined


def unresolved_imports(path: Path, notebook: str | None = None) -> list[str]:
    """Describe every contigo import of the notebook that does not resolve."""
    problems: list[str] = []
    for module, names in contigo_imports(path, notebook):
        spec = _find_spec(module)
        if spec is None:
            problems.append(f"{module}: module not found")
            continue
        if names == ("*",):
            continue
        for name in names:
            if not _name_resolves(module, name, spec):
                problems.append(f"{module}.{name}: name not defined")
    return problems


# ---------------------------------------------------------------------------
# NB-01: tests
# ---------------------------------------------------------------------------
def _write_notebook(tmp_path: Path, *cells: list[str]) -> Path:
    nb = {"cells": [{"cell_type": "code", "source": c} for c in cells]}
    path = tmp_path / "x.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def test_nb01_helpers_find_missing_names(tmp_path):
    """NB-01 helper: a bad name and a bad module are reported, good ones pass.

    Expected: ``Constellation`` and the submodule ``spacecraft`` resolve;
    ``ThirdBody`` (removed) and a nonexistent module do not.
    """
    path = _write_notebook(
        tmp_path,
        [
            "!pip install contigo\n",
            "%matplotlib inline\n",
            "from contigo.constellation import Constellation\n",
            "from contigo import spacecraft\n",
            "from contigo.forces.third_body_acc import ThirdBody, ThirdBodyEnv\n",
            "import contigo.nope\n",
        ],
    )
    nb = json.loads(path.read_text())
    nb["cells"].append(
        {"cell_type": "markdown", "source": ["from contigo.fake import Nothing"]}
    )
    path.write_text(json.dumps(nb), encoding="utf-8")

    assert unresolved_imports(path) == [
        "contigo.forces.third_body_acc.ThirdBody: name not defined",
        "contigo.nope: module not found",
    ]


def test_nb01_parses_multiline_imports_and_ipython_lines(tmp_path):
    """NB-01 helper: parenthesised and backslash-continued imports are parsed.

    Expected: both multi-line forms give all their names; ``!``/``%`` lines,
    indented ones and ``x = !cmd`` (also continued) become ``pass``; a ``%%``
    cell is skipped.
    """
    path = _write_notebook(
        tmp_path,
        [
            "from contigo.constellation import (\n",
            "    Constellation,\n",
            "    # a comment\n",
            "    Constellation as C2,\n",
            ")\n",
            "from contigo.spacecraft import \\\n",
            "    Spacecraft\n",
        ],
        [
            "try:\n",
            "    !pip install foo\n",
            "    out = !ls -l \\\n",
            "        /tmp\n",
            "except ImportError:\n",
            "    %pip install bar\n",
            "import contigo.config\n",
        ],
        ["%%bash\n", "from contigo.ignored import Nothing\n"],
    )
    assert contigo_imports(path) == [
        ("contigo.constellation", ("Constellation", "Constellation")),
        ("contigo.spacecraft", ("Spacecraft",)),
        ("contigo.config", ()),
    ]


def test_nb01_unparseable_cell_fails_with_notebook_and_cell(tmp_path):
    """NB-01 helper: a cell that still does not parse fails with a clear message."""
    path = _write_notebook(tmp_path, ["x = 1\n"], ["def broken(:\n", "    pass\n"])
    with pytest.raises(pytest.fail.Exception, match=r"nb\.ipynb: code cell 1"):
        contigo_imports(path, notebook="nb.ipynb")


_RESOLVER_PROBE = r"""
import json, sys
from pathlib import Path
import jpype

attempts = []
def guard(*a, **k):
    attempts.append("startJVM")
    raise RuntimeError("JVM start attempted")
jpype.startJVM = guard

sys.path.insert(0, sys.argv[1])
import test_package as tp

problems = tp.unresolved_imports(Path(sys.argv[2]))
executed = sorted(
    n for n, m in sys.modules.items()
    if (n == "contigo" or n.startswith("contigo.")) and not hasattr(m, "__path__")
)
print("RESULT:" + json.dumps({"problems": problems, "executed": executed,
                              "attempts": attempts, "jvm": jpype.isJVMStarted(),
                              "srp": "contigo.forces.srp_orekit" in sys.modules}))
"""


def test_nb01_resolver_never_executes_contigo_modules(tmp_path):
    """NB-01: resolving names does not import any contigo module or start the JVM.

    The resolver runs in a fresh interpreter with ``jpype.startJVM`` replaced
    by a raising guard, on a notebook that imports the JVM-starting
    ``srp_orekit`` and ``jb08_utils`` (names and as submodules), a star import
    chain and a missing name. Expected: all valid names resolve, the missing
    one is reported, no non-package contigo module is in ``sys.modules`` and
    the guard was never hit.
    """
    path = _write_notebook(
        tmp_path,
        [
            "from contigo.forces.srp_orekit import SRPOrekitCB\n",
            "from contigo.forces import srp_orekit, grav_pot\n",
            "from contigo.contigo_utils import jb08_utils\n",
            "from contigo.contigo_utils.jb08_utils import ThisDoesNotExist\n",
            "from contigo.edr_efd import EDRDensity\n",
        ],
    )
    result = _run_probe(_RESOLVER_PROBE, [str(TESTS_DIR), str(path)], tmp_path)
    assert result["problems"] == [
        "contigo.contigo_utils.jb08_utils.ThisDoesNotExist: name not defined"
    ]
    assert result["srp"] is False
    assert result["executed"] == []
    assert result["attempts"] == []
    assert result["jvm"] is False


# ---------------------------------------------------------------------------
# NB-01: _bound_names unit tests
# ---------------------------------------------------------------------------
def _bound(source: str, star: StarResolver | None = None) -> set[str]:
    return _bound_names(ast.parse(source).body, star)


def test_bound_names_definitions_and_assignments():
    """NB-01 helper: def/class/assign/annotated/augmented/import bind names.

    Attribute and subscript targets bind nothing; function bodies are skipped.
    """
    src = (
        "import os.path\nimport numpy as np\nfrom a import b, c as d\n"
        "def f():\n    hidden = 1\n"
        "class K:\n    inner = 1\n"
        "x, (y, *z) = 1, (2, 3)\n"
        "w: int = 1\nv += 1\n"
        "obj.attr = 1\ntab['k'] = 2\n"
    )
    assert _bound(src) == {
        "os",
        "np",
        "b",
        "d",
        "f",
        "K",
        "x",
        "y",
        "z",
        "w",
        "v",
    }


def test_bound_names_with_statement():
    """NB-01 helper: ``with`` binds its ``as`` targets and body (no ``orelse``)."""
    src = "with open('f') as fh, ctx() as (a, b), other():\n    inner = 1\n"
    assert _bound(src) == {"fh", "a", "b", "inner"}


def test_bound_names_for_while_if_with_orelse():
    """NB-01 helper: ``for`` target, and ``orelse`` of for/while/if are scanned."""
    src = (
        "for i, j in pairs():\n    k = 1\nelse:\n    z = 2\n"
        "while cond():\n    m = 1\nelse:\n    n = 2\n"
        "if c:\n    p = 1\nelif d:\n    q = 1\nelse:\n    r = 1\n"
    )
    assert _bound(src) == {"i", "j", "k", "z", "m", "n", "p", "q", "r"}


def test_bound_names_try_all_blocks():
    """NB-01 helper: try body, handler name and body, else and finally."""
    src = (
        "try:\n    a = 1\nexcept ValueError as err:\n    b = 1\n"
        "except KeyError:\n    c = 1\nelse:\n    d = 1\nfinally:\n    e = 1\n"
    )
    assert _bound(src) == {"a", "err", "b", "c", "d", "e"}


@pytest.mark.skipif(sys.version_info < (3, 11), reason="except* needs Python 3.11")
def test_bound_names_try_star():
    """NB-01 helper: ``try``/``except*`` is handled like ``try``."""
    src = "try:\n    a = 1\nexcept* ValueError as eg:\n    b = 1\n"
    assert _bound(src) == {"a", "eg", "b"}


def test_bound_names_match_captures():
    """NB-01 helper: ``match`` capture, star, mapping-rest and ``as`` names."""
    src = (
        "match cmd:\n"
        "    case [x, *rest]:\n        a = 1\n"
        "    case {'k': v, **kw}:\n        b = 1\n"
        "    case Point(px=py) as pt:\n        c = 1\n"
        "    case _:\n        d = 1\n"
    )
    assert _bound(src) == {"x", "rest", "a", "v", "kw", "b", "py", "pt", "c", "d"}


def test_bound_names_walrus():
    """NB-01 helper: walrus targets anywhere at module level, not in functions."""
    src = (
        "if (n := f()) > 1:\n    pass\n"
        "ys = [z for q in r if (z := q)]\n"
        "def g():\n    return (hidden := 1)\n"
    )
    assert _bound(src) == {"n", "ys", "z", "g"}


def test_bound_names_star_import_uses_resolver():
    """NB-01 helper: ``from x import *`` adds what the resolver returns.

    Without a resolver it adds nothing. The resolver gets the ``ImportFrom``
    node (relative level included).
    """
    seen = []

    def resolver(node: ast.ImportFrom) -> set[str]:
        seen.append((node.module, node.level))
        return {"alpha", "beta"}

    src = "from .sub import *\nfrom pkg import named\n"
    assert _bound(src) == {"named"}
    assert _bound(src, resolver) == {"alpha", "beta", "named"}
    assert seen == [("sub", 1)]


def test_name_resolves_through_star_import_one_level(monkeypatch):
    """NB-01: a name re-exported by a star import in a real module resolves.

    ``contigo.forces.third_body_acc`` binds ``ForceModel`` by an explicit
    import; the star resolver finds names in a contigo module by AST only.
    Earlier tests may already have imported the module, so it is removed from
    ``sys.modules`` for this test; the resolver must not import it again.
    """
    monkeypatch.delitem(sys.modules, "contigo.forces.base", raising=False)
    star = ast.parse("from contigo.forces.base import *").body[0]
    names = _star_names(star, "contigo.forces")
    assert "ForceModel" in names
    assert "contigo.forces.base" not in sys.modules


# ---------------------------------------------------------------------------
# NB-01: the notebooks
# ---------------------------------------------------------------------------
# Orekit_EDR.ipynb is not listed: it only uses the raw Orekit API and has no
# ``contigo`` import, so there is nothing to resolve. Add it back if it gains any.
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
    assert contigo_imports(path, notebook), f"{notebook} has no contigo imports"
    assert unresolved_imports(path, notebook) == []
