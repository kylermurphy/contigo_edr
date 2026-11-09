"""Fixtures for the plotting tests: a small three-spacecraft data set.

Constructing a ``Spacecraft`` loads the bundled SPICE leap-second kernel from
``contigo/data``; no network access is needed.

added: 09/11/2026 Claude Code
"""

import numpy as np
import pandas as pd
import pytest

from contigo.constellation import Constellation
from contigo.spacecraft import Spacecraft

IDS = ["A", "B", "C"]
N_PER_SC = 10


@pytest.fixture(scope="session")
def multi_sc() -> Spacecraft:
    """One ``Spacecraft`` holding IDs A, B, C with 10 points each."""
    t0 = pd.date_range("2020-01-01", periods=N_PER_SC, freq="10s")
    time = np.concatenate([t0.to_numpy() for _ in IDS])
    sc_id = np.repeat(IDS, N_PER_SC)

    state = np.zeros((len(IDS) * N_PER_SC, 6))
    for k in range(len(IDS)):
        sl = slice(k * N_PER_SC, (k + 1) * N_PER_SC)
        ang = np.linspace(0.0, 0.05, N_PER_SC) + 0.5 * k
        state[sl, 0] = 7.0e6 * np.cos(ang)
        state[sl, 1] = 7.0e6 * np.sin(ang)
        state[sl, 2] = 1.0e6 * (k + 1)
        state[sl, 3] = -7.5e3 * np.sin(ang)
        state[sl, 4] = 7.5e3 * np.cos(ang)
    return Spacecraft(
        state=state,
        time=time,
        sc_id_input=sc_id,
        tscale_input="GPS",
        cd=2.2,
        drag_area=1.0,
        sc_mass=100.0,
        cr=1.2,
        srp_area=1.0,
    )


@pytest.fixture(scope="session")
def constellation(multi_sc: Spacecraft) -> Constellation:
    """A ``Constellation`` built from ``multi_sc``."""
    return Constellation(spacecraft=multi_sc)
