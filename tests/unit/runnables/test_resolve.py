from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import Runnable, RunnableConfig

from zenpyre.runnables import resolve_runnable

MINIMAL_RUNNABLE_TARGET = "tests.unit.runnables.test_resolve.MinimalRunnable"


class MinimalRunnable(Runnable[Any, Any]):
    """Minimal concrete Runnable for testing."""

    def invoke(
        self,
        input: Any,  # noqa: A002
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Any:
        return input


##############################################
#        Tests for resolve_runnable          #
##############################################


# --- Pass-through ---


def test_resolve_runnable_returns_runnable_instance() -> None:
    assert isinstance(resolve_runnable(MinimalRunnable()), Runnable)


def test_resolve_runnable_passthrough_returns_same_instance() -> None:
    runnable = MinimalRunnable()
    assert resolve_runnable(runnable) is runnable


# --- From dict ---


def test_resolve_runnable_from_dict_returns_runnable() -> None:
    result = resolve_runnable({"_target_": MINIMAL_RUNNABLE_TARGET})
    assert isinstance(result, Runnable)


def test_resolve_runnable_from_dict_returns_correct_type() -> None:
    result = resolve_runnable({"_target_": MINIMAL_RUNNABLE_TARGET})
    assert isinstance(result, MinimalRunnable)


# --- Invalid input ---


def test_resolve_runnable_invalid_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a Runnable instance"):
        resolve_runnable("not-a-runnable")
