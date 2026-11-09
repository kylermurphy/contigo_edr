"""``ForceModel`` protocol conformance tests (PR T5, row FO-09).

Only models that can be built without the JVM or GMAT are used here:
``ThirdBodyEnv`` and ``EarthPotential`` (construction does not read the
``.gfc`` file). ``SRPOrekitCB`` starts the JVM at import (F4).

Note on F6: ``ThirdBodyEnv`` and ``EarthPotential`` *subclass* ``ForceModel``
explicitly, so ``isinstance`` is True for them through normal inheritance,
even though neither defines ``is_conservative`` (the Protocol only annotates
it). The structural check that F6 describes fails for a model that satisfies
the interface without subclassing, which is what new force models written
"to the protocol" (and the test doubles) do.

added: 09/11/2026 Claude Code
"""

from __future__ import annotations

import pytest
from _support import FakeForce, FakePotential

from contigo.forces.base import ForceModel
from contigo.forces.grav_pot import EarthPotential
from contigo.forces.third_body_acc import ThirdBodyEnv

MODELS = [ThirdBodyEnv, EarthPotential]


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_models_have_protocol_members(cls):
    """FO-09: each model has a ``name`` string and callable methods."""
    model = cls()
    assert isinstance(model.name, str) and model.name
    assert callable(model.acceleration)
    assert callable(model.potential)


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_subclassed_models_are_force_models(cls):
    """FO-09: explicit subclasses pass ``isinstance`` (nominal inheritance)."""
    assert isinstance(cls(), ForceModel)


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="F6/R6: ForceModel declares is_conservative which no model defines",
)
def test_models_define_declared_is_conservative(cls):
    """FO-09: every member the Protocol declares exists on the model."""
    assert isinstance(cls().is_conservative, bool)


@pytest.mark.parametrize("cls", [FakeForce, FakePotential], ids=lambda c: c.__name__)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F6/R6: ForceModel declares is_conservative which no model defines",
)
def test_structural_models_are_force_models(cls):
    """FO-09: a model with ``name``/``acceleration``/``potential`` but no
    ``ForceModel`` base class satisfies the runtime-checkable Protocol."""
    model = cls()
    assert callable(model.acceleration) and callable(model.potential)
    assert isinstance(model, ForceModel)
