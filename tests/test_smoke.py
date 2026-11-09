"""Smoke tests: the package and its subpackages import, and bundled data exists.

These run against whatever ``contigo`` is importable (usually the editable
source tree), so they do not prove the *wheel* is complete. The CI ``install``
job (task 0.5a, PK-02) builds and installs a non-editable wheel for that.

added: 09/11/2026 Claude Code
"""

import importlib
from pathlib import Path

import pytest

import contigo
import contigo.config as config

SUBPACKAGES = ["contigo.contigo_utils", "contigo.ephemeris", "contigo.forces"]

BUNDLED_FILES = [
    config.LEAP_FILE,
    config.PCK_FILE,
    "de440s.bsp",
    "EIGEN-2.gfc",
]


def test_import_contigo():
    assert contigo.__name__ == "contigo"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    assert importlib.import_module(name).__name__ == name


def test_data_dir_exists():
    assert Path(config.DATA_DIR).is_dir()


@pytest.mark.parametrize("fname", BUNDLED_FILES)
def test_bundled_data_present(fname):
    assert (Path(config.DATA_DIR) / fname).is_file()
