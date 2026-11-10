"""CONTIGO: Energy Dissipation Rate (EDR) and Effective Density (EFD) of
Earth-orbiting spacecraft.

Subpackages: ``contigo_utils`` (helpers), ``ephemeris`` (solar system ephemeris
providers) and ``forces`` (force models). This module is kept import-light;
submodules are imported on demand.

added: 03/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("contigo")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0+unknown"
