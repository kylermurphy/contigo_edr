"""Shared pytest configuration and fixtures for the contigo test suite (PR T0).

Tiers
-----
Tests marked ``network``, ``orekit`` or ``slow`` are skipped unless the matching
command-line flag is given (or ``--run-all``). Unmarked tests form the fast tier
that CI runs: they must not touch the network or start the JVM. The
``_no_network`` guard enforces this for contigo's own download helpers.

Fixtures (see docs/phase0_plan.md §4.0)
---------------------------------------
- ``circular_orbit``: factory for an exact circular Keplerian orbit in ECEF.
- ``sc_single`` / ``sc_multi`` / ``constellation_fixture``: synthetic objects.
- ``FakeProvider`` / ``FakePotential`` / ``FakeForce``: test doubles.
- ``esa_pod_small`` / ``orekit_ref_small`` / ``esa_constellation``: two orbits of
  real POD with the matching Orekit results (regenerate with
  ``tests/data/make_fixtures.py``).

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from _support import (  # noqa: E402
    ESA_PROPS,
    MULTI_MEMBERS,
    FakeProvider,
    load_bundled_kernels,
    make_circular_orbit,
    make_esa_constellation,
)

import contigo.config as config
from contigo.constellation import Constellation
from contigo.contigo_utils import utils
from contigo.spacecraft import Spacecraft

TIERS = ("network", "orekit", "slow")
DATA = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Tier options
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    for tier in TIERS:
        parser.addoption(
            f"--run-{tier}",
            action="store_true",
            default=False,
            help=f"run tests marked @pytest.mark.{tier}",
        )
    parser.addoption(
        "--run-all",
        action="store_true",
        default=False,
        help="run every test tier",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-all"):
        return
    for tier in TIERS:
        if config.getoption(f"--run-{tier}"):
            continue
        skip = pytest.mark.skip(reason=f"{tier} test: use --run-{tier} to run")
        for item in items:
            # check real markers only; item.keywords also holds node names
            if item.get_closest_marker(tier) is not None:
                item.add_marker(skip)


# ---------------------------------------------------------------------------
# Environment: bundled SPICE kernels, no network, isolated global state
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _isolated_cache_dir(tmp_path_factory: pytest.TempPathFactory):
    """Point ``CONTIGO_CACHE_DIR`` at a session temp dir.

    A developer's real per-user cache must never affect the tests.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CONTIGO_CACHE_DIR", str(tmp_path_factory.mktemp("contigo_cache")))
        yield


@pytest.fixture(scope="session", autouse=True)
def spice_offline(_isolated_cache_dir) -> None:
    """Load the bundled SPICE kernels and mark the NAIF download as done.

    ``SPICEEphem.load_kernels`` otherwise asks NAIF for file modification
    times on first use. Kernels are furnished with the same path strings that
    ``SPICEEphem`` builds, so it recognises them as already loaded.
    """
    load_bundled_kernels()


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Make contigo's download helpers fail loudly outside the network tier."""
    if request.node.get_closest_marker("network") is not None:
        return

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "network access in a fast-tier test; mark it @pytest.mark.network"
        )

    monkeypatch.setattr(utils, "dl_file", _blocked)
    monkeypatch.setattr(utils, "wf_mtime", _blocked)


@pytest.fixture(autouse=True)
def restore_config_state():
    """Snapshot and restore ``contigo.config.state`` around every test.

    The potential-coefficient cache and the loaded flags otherwise leak
    between tests.
    """
    saved = dict(config.state)
    yield
    config.state.clear()
    config.state.update(saved)


# ---------------------------------------------------------------------------
# Synthetic orbits
# ---------------------------------------------------------------------------
@pytest.fixture
def circular_orbit() -> Callable[..., tuple[np.ndarray, pd.DatetimeIndex]]:
    """Factory fixture: ``circular_orbit(alt=..., inc_deg=..., n=..., dt=...)``."""
    return make_circular_orbit


@pytest.fixture
def sc_single() -> Spacecraft:
    """One spacecraft, 400 km circular orbit, 600 samples at 10 s, GPS time."""
    state, time = make_circular_orbit()
    return Spacecraft(
        state=state,
        time=time,
        sc_id_input=np.full(len(time), "SAT-A"),
        tscale_input="GPS",
        **ESA_PROPS,
    )


@pytest.fixture
def sc_multi() -> Spacecraft:
    """Three IDs in one Spacecraft, rows shuffled, per-row properties.

    Each member has 200 samples at 10 s; ``cd`` differs per member so slicing
    errors in ``split_by_id`` are visible.
    """
    states, times, ids, cds = [], [], [], []
    for sc_id, alt, inc, cd in MULTI_MEMBERS:
        s, t = make_circular_orbit(alt=alt, inc_deg=inc, n=200)
        states.append(s)
        times.append(t)
        ids.append(np.full(len(t), sc_id))
        cds.append(np.full(len(t), cd))
    state = np.vstack(states)
    time = pd.DatetimeIndex(np.concatenate([t.values for t in times]))
    sc_id = np.concatenate(ids)
    cd = np.concatenate(cds)
    order = np.random.default_rng(17).permutation(len(sc_id))
    return Spacecraft(
        state=state[order],
        time=time[order],
        sc_id_input=sc_id[order],
        tscale_input="GPS",
        cd=cd[order],
        drag_area=1.1,
        sc_mass=430.0,
        cr=1.8,
        srp_area=1.0,
    )


@pytest.fixture
def constellation_fixture(sc_multi: Spacecraft) -> Constellation:
    """``Constellation`` of the three ``sc_multi`` members."""
    return Constellation(spacecraft=sc_multi)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


# ---------------------------------------------------------------------------
# Real reference data (two orbits)
# ---------------------------------------------------------------------------
def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / name, float_precision="round_trip")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df


@pytest.fixture(scope="session")
def esa_pod_small() -> pd.DataFrame:
    """First 1104 rows of ``data/ESA_pod.hdf``: ECEF **km** and km/s, GPS time."""
    return _read("esa_pod_small.csv.gz")


@pytest.fixture(scope="session")
def orekit_ref_small() -> pd.DataFrame:
    """Orekit results for the ``esa_pod_small`` epochs, in **SI** units."""
    return _read("orekit_ref_small.csv.gz")


@pytest.fixture
def esa_constellation(esa_pod_small: pd.DataFrame) -> Constellation:
    """``Constellation`` with ID ``'ESA'`` built from ``esa_pod_small`` in SI.

    Uses ``unit_input='km'`` (the raw file is in km) and the ESA properties of
    ``CONTIGO_Colab.ipynb``: Cd 3.5, drag area 1.1 m^2, mass 430 kg, Cr 1.8,
    SRP area 1 m^2.
    """
    return make_esa_constellation(esa_pod_small)
