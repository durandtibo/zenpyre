from __future__ import annotations

from typing import Any

import pytest

from zenpyre.utils.config import MISSING, BaseConfig
from zenpyre.utils.resolve import resolve_object


class Animal:
    """Base class for testing."""


class Dog(Animal):
    """Concrete subclass for testing."""

    def __init__(self, name: str = "Rex") -> None:
        self.name = name


class DogConfig(BaseConfig):
    """A BaseConfig that resolves to a Dog for testing."""

    def __init__(self, name: str = "Rex") -> None:
        self.name = name

    def get_value(self, name: str, default: Any = MISSING) -> Any:
        kwargs = self.to_kwargs()
        if name in kwargs:
            return kwargs[name]
        if default is not MISSING:
            return default
        raise KeyError(name)

    def to_kwargs(self) -> dict[str, Any]:
        return {"_target_": "tests.unit.utils.test_resolve.Dog", "name": self.name}


####################################
#     Tests for resolve_object     #
####################################


# --- Pass-through ---


def test_resolve_object_returns_same_instance_when_no_cls() -> None:
    obj = Dog()
    assert resolve_object(obj) is obj


def test_resolve_object_returns_same_instance_with_matching_cls() -> None:
    obj = Dog()
    assert resolve_object(obj, cls=Dog) is obj


def test_resolve_object_returns_same_instance_with_parent_cls() -> None:
    obj = Dog()
    assert resolve_object(obj, cls=Animal) is obj


# --- Default cls=object ---


def test_resolve_object_default_cls_accepts_any_instance() -> None:
    assert resolve_object(42) == 42


def test_resolve_object_default_cls_accepts_string() -> None:
    assert resolve_object("hello") == "hello"


# --- From dict ---


def test_resolve_object_from_dict_returns_instance() -> None:
    result = resolve_object({"_target_": "tests.unit.utils.test_resolve.Dog"}, cls=Dog)
    assert isinstance(result, Dog)


def test_resolve_object_from_dict_no_cls_still_resolves() -> None:
    result = resolve_object({"_target_": "tests.unit.utils.test_resolve.Dog"})
    assert isinstance(result, Dog)


# --- From BaseConfig ---


def test_resolve_object_from_base_config_returns_instance() -> None:
    result = resolve_object(DogConfig(), cls=Dog)
    assert isinstance(result, Dog)


def test_resolve_object_from_base_config_no_cls_still_resolves() -> None:
    result = resolve_object(DogConfig())
    assert isinstance(result, Dog)


def test_resolve_object_from_base_config_uses_config_fields() -> None:
    result = resolve_object(DogConfig(name="Fido"), cls=Dog)
    assert result.name == "Fido"


def test_resolve_object_from_base_config_wrong_cls_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a str instance"):
        resolve_object(DogConfig(), cls=str)


# --- Invalid input ---


def test_resolve_object_invalid_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a Animal instance"):
        resolve_object("not-an-animal", cls=Animal)


def test_resolve_object_wrong_subclass_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a Dog instance"):
        resolve_object(Animal(), cls=Dog)


def test_resolve_object_from_dict_wrong_cls_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a str instance"):
        resolve_object({"_target_": "tests.unit.utils.test_resolve.Dog"}, cls=str)
