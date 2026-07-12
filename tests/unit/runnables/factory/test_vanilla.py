from __future__ import annotations

from typing import Any

from coola.equality import objects_are_equal
from langchain_core.runnables import Runnable, RunnableConfig

from zenpyre.runnables.factory import BaseRunnableFactory, RunnableFactory


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


##############################################
#     Tests for RunnableFactory              #
##############################################


# --- Inheritance ---


def test_runnable_factory_is_base_runnable_factory() -> None:
    assert isinstance(RunnableFactory(_make_runnable()), BaseRunnableFactory)


# --- make_runnable ---


def test_runnable_factory_make_runnable_returns_runnable() -> None:
    factory = RunnableFactory(_make_runnable())
    assert isinstance(factory.make_runnable(), Runnable)


def test_runnable_factory_make_runnable_returns_same_instance() -> None:
    runnable = _make_runnable()
    factory = RunnableFactory(runnable)
    assert factory.make_runnable() is runnable


def test_runnable_factory_make_runnable_returns_same_instance_across_calls() -> None:
    runnable = _make_runnable()
    factory = RunnableFactory(runnable)
    assert factory.make_runnable() is factory.make_runnable()


# --- _get_repr_kwargs ---


def test_runnable_factory_get_repr_kwargs() -> None:
    runnable = _make_runnable()
    factory = RunnableFactory(runnable)
    assert objects_are_equal(factory._get_repr_kwargs(), {"runnable": runnable})


# --- __repr__ and __str__ ---


def test_runnable_factory_repr_starts_with_class_name() -> None:
    factory = RunnableFactory(_make_runnable())
    assert repr(factory).startswith("RunnableFactory(")


def test_runnable_factory_str_starts_with_class_name() -> None:
    factory = RunnableFactory(_make_runnable())
    assert str(factory).startswith("RunnableFactory(")


def test_runnable_factory_repr_contains_runnable() -> None:
    factory = RunnableFactory(_make_runnable())
    assert "runnable" in repr(factory)


def test_runnable_factory_str_contains_runnable() -> None:
    factory = RunnableFactory(_make_runnable())
    assert "runnable" in str(factory)
