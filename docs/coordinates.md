# Coordinates and Earth orientation

`contigo.contigo_utils.coords` converts the Earth-fixed (ECEF) states that `Spacecraft`
stores into geodetic coordinates and into the inertial J2000 frame. Units are metres, m/s and
**radians** throughout.

| Function | In | Out | Needs |
|----------|----|-----|-------|
| `ecef_to_geodetic(r_ecef, n_iter=5)` | `(..., 3)` ECEF position | `(..., 3)` `[lat, lon, alt]` (WGS-84) | NumPy only |
| `geodetic_to_ecef(llh)` | `(..., 3)` `[lat, lon, alt]` | `(..., 3)` ECEF position | NumPy only |
| `ecef_to_inertial(state_ecef, et, *, fallback="warn")` | `(N, 3)` positions or `(N, 6)` states, `(N,)` ET | same shape, J2000 | SPICE kernels (bundled) |
| `inertial_to_ecef(state_inertial, et, *, fallback="warn")` | same, J2000 | same shape, ECEF | SPICE kernels (bundled) |

## Geodetic coordinates

```python
import numpy as np
from contigo.contigo_utils.coords import ecef_to_geodetic, geodetic_to_ecef

llh = ecef_to_geodetic(sc.state_ecef[:, :3])   # (N, 3)
lat_deg = np.degrees(llh[:, 0])
lon_deg = np.degrees(llh[:, 1])                 # in [-180, 180]
alt_km = llh[:, 2] / 1e3                        # height above the WGS-84 ellipsoid
```

- Latitude is **geodetic** (the angle of the ellipsoid normal), not geocentric.
  `Spacecraft.spherical()` gives geocentric latitude and the distance from Earth's centre,
  which is *not* altitude.
- The conversion is vectorised: a million points take about 0.2 s, and the round trip
  `geodetic_to_ecef(ecef_to_geodetic(r))` is exact to better than 1e-7 m up to 40,000 km
  altitude. It is well behaved at the poles.
- The WGS-84 constants are `WGS84_A`, `WGS84_F` and `WGS84_E2` in
  `contigo.contigo_utils.constants`.

## Inertial (J2000) frame

```python
from contigo.contigo_utils.coords import ecef_to_inertial, inertial_to_ecef

state_j2000 = ecef_to_inertial(sc.state_ecef, sc.sspice_et)     # (N, 6) -> (N, 6)
r_j2000 = ecef_to_inertial(sc.state_ecef[:, :3], sc.sspice_et)  # (N, 3) -> (N, 3)
```

- `et` is SPICE ephemeris time (TDB seconds past J2000), the same as `Spacecraft.sspice_et`,
  one per row.
- The transform is SPICE's `ITRF93` to `J2000` (`sxform` for states, `pxform` for positions).
  For states it includes Earth's rotation in the velocity:
  `v_inertial = R (v_ecef + omega x r_ecef)`.
- Accuracy: against Orekit's inertial states for the ESA reference orbit, the difference is
  under 1 m and 1 mm/s.
- The kernels it needs are loaded automatically (by `kernels.check_pck`); you do not need a
  `SPICEEphem`.
- **It loops over epochs in Python (~5 µs each).** That is 0.1 s for 20,000 epochs. For plots,
  thin the data first (for example with `contigo.plotting._util.decimate`).

## Earth orientation coverage

The high-precision transform needs Earth orientation data (`earth_latest_high_prec.bpc`) for
your dates. The bundled copy covers **2000-01-01 to 2026-05-30**. For other dates,
`ecef_to_inertial`:

1. checks whether the loaded kernels cover all of `et`; if so, uses `ITRF93`;
2. otherwise tries once per session to download the newest kernel from NAIF into the cache
   (not when `CONTIGO_OFFLINE=1`, and not if `SPICEEphem` already checked NAIF this session);
3. if the dates are still not covered:
   - `fallback="warn"` (default): issues a `RuntimeWarning` and uses the low-precision
     `IAU_EARTH` frame. This ignores nutation and polar motion: **positions are off by about
     10 km** at LEO (about 0.08 degrees of longitude). Fine for a plot, not for physics.
   - `fallback="raise"`: raises `RuntimeError`, naming the dates the loaded kernel covers. Use
     this in EDR or validation code.

Check coverage yourself with:

```python
from contigo.contigo_utils import kernels
kernels.check_pck()                       # load the kernels
kernels.pck_covers(et.min(), et.max())    # True if ITRF93 data covers the range
```

NAIF's kernel also contains a few months of predictions, so a download covers up to
"recently". For data from the last few weeks, or offline work on recent data, expect the
fallback.

## Picking dates for examples and tests

Choose example and test data **before 2026-05-30** (inside the bundled coverage), so it works
offline, on CI and on a Colab cold start without a download or a 10 km warning. Tests that
need uncovered dates should set `CONTIGO_OFFLINE=1` and expect the warning (see
`tests/test_coords.py`, FX-06 to FX-08).
