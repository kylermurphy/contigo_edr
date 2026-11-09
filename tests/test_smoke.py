"""Smoke tests: the package and its subpackages import, and bundled data ships.

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
    # a non-editable install with packages=["contigo"] used to drop these
    assert importlib.import_module(name).__name__ == name


def test_data_dir_exists():
    assert Path(config.DATA_DIR).is_dir()


@pytest.mark.parametrize("fname", BUNDLED_FILES)
def test_bundled_data_present(fname):
    assert (Path(config.DATA_DIR) / fname).is_file()
