"""Plotting tools for CONTIGO spacecraft and constellations.

Matplotlib is an optional dependency (``pip install 'contigo[plot]'``) and is
imported lazily inside the plotting modules, so ``import contigo.plotting``
works without it. The public plot functions are added in Phase 1; until then
``__all__`` is empty.

Private helpers:

- ``contigo.plotting._util``: input normalisation, decimation and
  antimeridian splitting (NumPy only).
- ``contigo.plotting._style``: lazy matplotlib import, per-spacecraft colours
  and the shared rc style.

added: 09/11/2026 Claude Code
"""

__all__: list[str] = []
