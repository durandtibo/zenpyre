from __future__ import annotations

from typing import Any

from coola.equality import objects_are_equal
from langchain_core.runnables import Runnable, RunnableConfig

from zenpyre.runnables.factory import (
    BaseRunnableFactory,
    ConfigurableRunnableFactory,
)

MINIMAL_RUNNABLE_TARGET = "tests.unit.runnables.factory.test_configurable.MinimalRunnable"


class MinimalRunnable(Runnable[Any, Any]):
    """Minimal concrete Runnable for testing."""

    def invoke(
        self,
        input: Any,  # noqa: A002
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Any:
        return input


def _make_runnable() -> MinimalRunnable:
    """Return a MinimalRunnable instance for testing."""
    return MinimalRunnable()


##################################################
#     Tests for ConfigurableRunnableFactory      #
##################################################


# --- Inheritance ---


def test_configurable_runnable_factory_is_base_runnable_factory() -> None:
    assert isinstance(ConfigurableRunnableFactory(_make_runnable()), BaseRunnableFactory)


# --- make_runnable from instance ---


def test_configurable_runnable_factory_make_runnable_returns_runnable() -> None:
    factory = ConfigurableRunnableFactory(_make_runnable())
    assert isinstance(factory.make_runnable(), Runnable)


def test_configurable_runnable_factory_make_runnable_returns_same_instance() -> None:
    runnable = _make_runnable()
    factory = ConfigurableRunnableFactory(runnable)
    assert factory.make_runnable() is runnable


# --- make_runnable from dict ---


def test_configurable_runnable_factory_make_runnable_from_dict_returns_runnable() -> None:
    factory = ConfigurableRunnableFactory({"_target_": MINIMAL_RUNNABLE_TARGET})
    assert isinstance(factory.make_runnable(), Runnable)


def test_configurable_runnable_factory_make_runnable_from_dict_returns_correct_type() -> None:
    factory = ConfigurableRunnableFactory({"_target_": MINIMAL_RUNNABLE_TARGET})
    assert isinstance(factory.make_runnable(), MinimalRunnable)


# --- _get_repr_kwargs ---


def test_configurable_runnable_factory_get_repr_kwargs_instance() -> None:
    runnable = _make_runnable()
    factory = ConfigurableRunnableFactory(runnable)
    assert objects_are_equal(factory._get_repr_kwargs(), {"runnable": runnable})


def test_configurable_runnable_factory_get_repr_kwargs_dict_input() -> None:
    config = {"_target_": MINIMAL_RUNNABLE_TARGET}
    factory = ConfigurableRunnableFactory(config)
    assert objects_are_equal(factory._get_repr_kwargs(), {"runnable": config})


# --- __repr__ and __str__ ---


def test_configurable_runnable_factory_repr_starts_with_class_name() -> None:
    factory = ConfigurableRunnableFactory(_make_runnable())
    assert repr(factory).startswith("ConfigurableRunnableFactory(")


def test_configurable_runnable_factory_str_starts_with_class_name() -> None:
    factory = ConfigurableRunnableFactory(_make_runnable())
    assert str(factory).startswith("ConfigurableRunnableFactory(")


def test_configurable_runnable_factory_repr_contains_runnable() -> None:
    factory = ConfigurableRunnableFactory(_make_runnable())
    assert "runnable" in repr(factory)


def test_configurable_runnable_factory_str_contains_runnable() -> None:
    factory = ConfigurableRunnableFactory(_make_runnable())
    assert "runnable" in str(factory)
