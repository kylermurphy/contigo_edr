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
DOUBLES = [FakeForce, FakePotential]


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_models_have_nonempty_name(cls):
    """FO-09: each model has a non-empty ``str`` name."""
    name = cls().name
    assert isinstance(name, str) and name.strip()


def test_third_body_env_potential_not_implemented(constellation_fixture):
    """FO-09: ``ThirdBodyEnv`` is acceleration-only; ``potential`` raises."""
    with pytest.raises(NotImplementedError):
        ThirdBodyEnv().potential(constellation_fixture)


def test_earth_potential_acceleration_not_implemented(constellation_fixture):
    """FO-09: ``EarthPotential`` is potential-only; ``acceleration`` raises."""
    with pytest.raises(NotImplementedError):
        EarthPotential().acceleration(constellation_fixture, None)


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


@pytest.mark.parametrize("cls", DOUBLES, ids=lambda c: c.__name__)
def test_doubles_implement_protocol_methods(cls):
    """FO-09: precondition for the structural check below: the doubles have
    a ``name`` and callable ``acceleration`` and ``potential``."""
    model = cls()
    assert isinstance(model.name, str) and model.name
    assert callable(model.acceleration) and callable(model.potential)


@pytest.mark.parametrize("cls", DOUBLES, ids=lambda c: c.__name__)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F6/R6: ForceModel declares is_conservative which no model defines",
)
def test_structural_models_are_force_models(cls):
    """FO-09: a model with ``name``/``acceleration``/``potential`` but no
    ``ForceModel`` base class satisfies the runtime-checkable Protocol."""
    assert isinstance(cls(), ForceModel)
