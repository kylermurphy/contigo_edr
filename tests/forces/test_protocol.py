"""``ForceModel`` protocol conformance tests (PR T5, row FO-09).

Only models that can be built without the JVM or GMAT are used here:
``ThirdBodyEnv`` and ``EarthPotential`` (construction does not read the
``.gfc`` file). ``SRPOrekitCB`` starts the JVM at import (F4).

Note on F6/R6: ``ThirdBodyEnv`` and ``EarthPotential`` *subclass*
``ForceModel`` explicitly, so ``isinstance`` is True for them through normal
inheritance. The Protocol used to declare ``is_conservative``, which no model
defined, so duck-typed models (and the test doubles) failed the structural
check. R6 removed the member; the structural check now passes.

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


def test_protocol_does_not_declare_is_conservative():
    """FO-09/R6: ``is_conservative`` is no longer part of the Protocol; the
    only declared data member is ``name``."""
    assert "is_conservative" not in getattr(ForceModel, "__annotations__", {})
    assert "is_conservative" not in getattr(ForceModel, "__protocol_attrs__", set())
    assert getattr(ForceModel, "__annotations__", {}) == {"name": str}


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_models_define_declared_members(cls):
    """FO-09: every member the Protocol declares exists on the model."""
    model = cls()
    assert isinstance(model.name, str)
    assert callable(model.acceleration) and callable(model.potential)


@pytest.mark.parametrize("cls", DOUBLES, ids=lambda c: c.__name__)
def test_doubles_implement_protocol_methods(cls):
    """FO-09: precondition for the structural check below: the doubles have
    a ``name`` and callable ``acceleration`` and ``potential``."""
    model = cls()
    assert isinstance(model.name, str) and model.name
    assert callable(model.acceleration) and callable(model.potential)


@pytest.mark.parametrize("cls", DOUBLES, ids=lambda c: c.__name__)
def test_structural_models_are_force_models(cls):
    """FO-09: a model with ``name``/``acceleration``/``potential`` but no
    ``ForceModel`` base class satisfies the runtime-checkable Protocol."""
    assert isinstance(cls(), ForceModel)
