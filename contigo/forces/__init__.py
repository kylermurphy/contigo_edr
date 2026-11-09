"""Force models used in the EDR computation.

Each model implements the ``ForceModel`` protocol (``forces/base.py``).

- ``ThirdBodyEnv``: third-body (Sun, Moon, planets) acceleration.
- ``EarthPotential``: Earth gravity potential and acceleration from a spherical
  harmonic (ICGEM ``.gfc``) gravity field.
- ``SRPOrekitCB``: cannonball solar radiation pressure using Orekit.
- ``SRPAcc``: solar radiation pressure using GMAT (may be removed).

added: 26/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
