# Writing a force model

CONTIGO computes the energy dissipation rate (EDR) by integrating the work done by every
modelled force along each orbit (equations A18 and A19 of
[doi:10.1029/2024EA003898](https://doi.org/10.1029/2024EA003898)). Each force
is a class that follows the `ForceModel` protocol in `contigo/forces/base.py`. Writing such a
class is all you need to do to add a force to the calculation.

## The protocol

```python
class ForceModel(Protocol):
    name: str

    def acceleration(self, constellation, solarsys_env) -> dict[sc_id, ndarray]: ...
    def potential(self, constellation) -> dict[sc_id, ndarray]: ...
```

| Member | Contract |
|--------|----------|
| `name` | A short label. It keys the force's result inside `EDRDensity`; if two models share a name, the later ones get a numbered key and a warning is logged |
| `acceleration` | Return `{sc_id: array}` with one entry per spacecraft in `constellation.spacecraft`. Each array is `(N, 3)`, or `(n_bodies, N, 3)` for a multi-body force (summed over bodies before integration). **ECEF, m/s²**, rows in the same order as `sc.state_ecef` |
| `potential` | Only for the gravity potential passed as `potential_model` (`EarthPotential`): `{sc_id: (N,)}` in m²/s². Other forces raise `NotImplementedError` |

What a force can use, per spacecraft `sc` (a single-ID `Spacecraft`):

- `sc.state_ecef` `(N, 6)`: position (m) and velocity (m/s) in the rotating Earth-fixed frame.
- `sc.sspice_et`, `sc.sspice_gps`, `sc.sc_utc`: the three time axes.
- `sc.cd_arr`, `sc.drag_area_arr`, `sc.sc_mass_arr`, `sc.cr_arr`, `sc.srp_area_arr`: physical
  properties as `(N,)` arrays (NaN if not given).
- `solarsys_env.get_ephem(ephem_time=..., gps_time=..., utc_time=...)`: cached Sun, Moon and
  planet positions (`ThirdBodyEnv` shows how). `solarsys_env.GM` has their GM values.

## Example: drag in a constant-density atmosphere

A toy model, to show the shape of a force. It assumes the atmosphere rotates with the Earth,
so the ECEF velocity is the velocity relative to the air.

```python
import numpy as np
import numpy.typing as npt

from contigo.constellation import Constellation
from contigo.forces.base import ForceModel
from contigo.solar_system_ephem import SolarSystemEnvironment


class ConstantDensityDrag(ForceModel):
    """a = -1/2 * rho * Cd * A / m * |v| * v, with v the Earth-fixed velocity."""

    name = "ConstantDensityDrag"

    def __init__(self, rho: float = 1e-12):  # kg/m^3
        self.rho = rho

    def acceleration(
        self, constellation: Constellation, solarsys_env: SolarSystemEnvironment
    ) -> dict[str, npt.NDArray[np.float64]]:
        acc = {}
        for sc_id, sc in constellation.spacecraft.items():
            v = sc.state_ecef[:, 3:6]                                     # (N, 3) m/s
            speed = np.linalg.norm(v, axis=1, keepdims=True)              # (N, 1)
            b = (sc.cd_arr * sc.drag_area_arr / sc.sc_mass_arr)[:, None]  # (N, 1) m^2/kg
            acc[sc_id] = -0.5 * self.rho * b * speed * v                  # (N, 3) m/s^2
        return acc

    def potential(self, constellation: Constellation):
        raise NotImplementedError("drag is not conservative")
```

Subclassing `ForceModel` is optional (the protocol is structural), but it documents intent and
`isinstance(model, ForceModel)` is then True.

Use it next to the built-in forces:

```python
from contigo.edr_efd import EDRDensity
from contigo.ephemeris.spice_ephem import SPICEEphem
from contigo.forces.grav_pot import EarthPotential
from contigo.forces.third_body_acc import ThirdBodyEnv
from contigo.solar_system_ephem import SolarSystemEnvironment

env = SolarSystemEnvironment(bodies=["SUN", "MOON"], tolerance=0.01, provider=SPICEEphem())
edr = EDRDensity(
    constellation=constellation,
    solarsys_env=env,
    force_models=[ThirdBodyEnv(), ConstantDensityDrag(rho=1e-12)],
    potential_model=EarthPotential(lmax=40),
)
edr_by_id = edr.compute_edr()   # {sc_id: (N,) EDR in m^2/s^2}
```

## Built-in forces

| Class | Module | Notes |
|-------|--------|-------|
| `ThirdBodyEnv` | `forces/third_body_acc.py` | Sun, Moon and planets from the `SolarSystemEnvironment`; returns `(n_bodies, N, 3)` |
| `SRPOrekitCB` | `forces/srp_orekit.py` | Cannonball solar radiation pressure via Orekit (starts the JVM on first use) |
| `EarthPotential` | `forces/grav_pot.py` | Spherical-harmonic gravity potential from an ICGEM `.gfc` file; pass it as `potential_model`, not in `force_models` |
| `SRPAcc` | `forces/srp_gmat.py` | Solar radiation pressure via GMAT. Likely to be removed (see `docs/ToDo.md`) |
| `ThirdBodyAcc` | `forces/third_body_acc.py` | **Deprecated**, removed in 0.3.0; use `ThirdBodyEnv` |

## Testing a force

Test the numbers, not just the shapes. `tests/_support.py` has helpers that need no data or
network:

- `make_circular_orbit(...)`: an exact circular orbit, already in ECEF;
- `FakeProvider`: an ephemeris provider that counts calls;
- `FakePotential` (`GM/r`) and `FakeForce` (`a = -k v`), whose EDR is known analytically.

For example, for an equatorial circular orbit with `FakeForce`, `EDR(t) = k |v|² t`
(`tests/test_edr_efd.py`, ED-02). For a new force, compare against a hand calculation at a
few points, and against an independent tool (Orekit, GMAT) where one exists; the stored
Orekit results in `tests/data/orekit_ref_small.csv.gz` are an example. See
[testing.md](testing.md) for the test tiers and conventions.
