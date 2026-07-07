from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from zenpyre.runnables import InputOutputPair, InputOutputRunnable


class _TrackingRunnable(Runnable[str, str]):
    """A runnable that records how it was called (invoke vs.

    batch vs. ainvoke vs. abatch), so tests can verify
    InputOutputRunnable delegates to the right method on the wrapped
    runnable instead of silently falling back to Runnable's default per-
    item implementation.
    """

    def __init__(self) -> None:
        self.invoke_calls: list[str] = []
        self.batch_calls: list[list[str]] = []
        self.ainvoke_calls: list[str] = []
        self.abatch_calls: list[list[str]] = []

    def invoke(
        self,
        input: str,  # noqa: A002
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        self.invoke_calls.append(input)
        return input.upper()

    def batch(
        self,
        inputs: list[str],
        config: RunnableConfig | list[RunnableConfig] | None = None,  # noqa: ARG002
        *,
        return_exceptions: bool = False,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> list[str]:
        self.batch_calls.append(list(inputs))
        return [x.upper() for x in inputs]

    async def ainvoke(
        self,
        input: str,  # noqa: A002
        config: RunnableConfig | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        self.ainvoke_calls.append(input)
        return input.upper()

    async def abatch(
        self,
        inputs: list[str],
        config: RunnableConfig | list[RunnableConfig] | None = None,  # noqa: ARG002
        *,
        return_exceptions: bool = False,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> list[str]:
        self.abatch_calls.append(list(inputs))
        return [x.upper() for x in inputs]


def _failing_lambda(fail_on: str) -> RunnableLambda:
    def fn(x: str) -> str:
        if x == fail_on:
            msg = f"failed for {x}"
            raise ValueError(msg)
        return x.upper()

    return RunnableLambda(fn)


####################################
#     Tests for InputOutputPair     #
####################################


def test_input_output_pair_stores_fields() -> None:
    pair = InputOutputPair(input="hello", output="HELLO")
    assert pair.input == "hello"
    assert pair.output == "HELLO"


def test_input_output_pair_is_frozen() -> None:
    pair = InputOutputPair(input="hello", output="HELLO")
    with pytest.raises(FrozenInstanceError):
        pair.output = "CHANGED"


def test_input_output_pair_equality() -> None:
    pair_a = InputOutputPair(input="hello", output="HELLO")
    pair_b = InputOutputPair(input="hello", output="HELLO")
    assert pair_a == pair_b


def test_input_output_pair_inequality_different_output() -> None:
    pair_a = InputOutputPair(input="hello", output="HELLO")
    pair_b = InputOutputPair(input="hello", output="WORLD")
    assert pair_a != pair_b


########################################
#     Tests for InputOutputRunnable     #
########################################


# --- invoke ---


def test_input_output_runnable_invoke_returns_input_output_pair() -> None:
    wrapped = InputOutputRunnable(RunnableLambda(lambda x: x.upper()))
    result = wrapped.invoke("hello")
    assert isinstance(result, InputOutputPair)


def test_input_output_runnable_invoke_pairs_input_and_output() -> None:
    wrapped = InputOutputRunnable(RunnableLambda(lambda x: x.upper()))
    result = wrapped.invoke("hello")
    assert result == InputOutputPair(input="hello", output="HELLO")


def test_input_output_runnable_invoke_propagates_exception() -> None:
    wrapped = InputOutputRunnable(_failing_lambda(fail_on="hello"))
    with pytest.raises(ValueError, match="failed for hello"):
        wrapped.invoke("hello")


# --- ainvoke ---


def test_input_output_runnable_ainvoke_returns_input_output_pair() -> None:
    wrapped = InputOutputRunnable(RunnableLambda(lambda x: x.upper()))
    result = asyncio.run(wrapped.ainvoke("hello"))
    assert isinstance(result, InputOutputPair)


def test_input_output_runnable_ainvoke_pairs_input_and_output() -> None:
    wrapped = InputOutputRunnable(RunnableLambda(lambda x: x.upper()))
    result = asyncio.run(wrapped.ainvoke("hello"))
    assert result == InputOutputPair(input="hello", output="HELLO")


def test_input_output_runnable_ainvoke_uses_inner_ainvoke() -> None:
    inner = _TrackingRunnable()
    wrapped = InputOutputRunnable(inner)
    asyncio.run(wrapped.ainvoke("hello"))
    assert inner.ainvoke_calls == ["hello"]
    assert inner.invoke_calls == []


def test_input_output_runnable_ainvoke_propagates_exception() -> None:
    wrapped = InputOutputRunnable(_failing_lambda(fail_on="hello"))
    with pytest.raises(ValueError, match="failed for hello"):
        asyncio.run(wrapped.ainvoke("hello"))


# --- batch ---


def test_input_output_runnable_batch_returns_list_of_input_output_pairs() -> None:
    wrapped = InputOutputRunnable(RunnableLambda(lambda x: x.upper()))
    results = wrapped.batch(["a", "b", "c"])
    assert all(isinstance(r, InputOutputPair) for r in results)


def test_input_output_runnable_batch_pairs_inputs_and_outputs_in_order() -> None:
    wrapped = InputOutputRunnable(RunnableLambda(lambda x: x.upper()))
    results = wrapped.batch(["a", "b", "c"])
    assert results == [
        InputOutputPair(input="a", output="A"),
        InputOutputPair(input="b", output="B"),
        InputOutputPair(input="c", output="C"),
    ]


def test_input_output_runnable_batch_empty_list_returns_empty_list() -> None:
    wrapped = InputOutputRunnable(RunnableLambda(lambda x: x.upper()))
    assert wrapped.batch([]) == []


def test_input_output_runnable_batch_uses_inner_batch_not_inner_invoke() -> None:
    # This is the key behavior distinguishing our override from the
    # default Runnable.batch (which would call inner.invoke once per
    # item via a thread pool instead of inner.batch once).
    inner = _TrackingRunnable()
    wrapped = InputOutputRunnable(inner)
    wrapped.batch(["a", "b", "c"])
    assert inner.batch_calls == [["a", "b", "c"]]
    assert inner.invoke_calls == []


def test_input_output_runnable_batch_raises_by_default_on_failure() -> None:
    wrapped = InputOutputRunnable(_failing_lambda(fail_on="b"))
    with pytest.raises(ValueError, match="failed for b"):
        wrapped.batch(["a", "b", "c"])


def test_input_output_runnable_batch_return_exceptions_keeps_raw_exception() -> None:
    wrapped = InputOutputRunnable(_failing_lambda(fail_on="b"))
    results = wrapped.batch(["a", "b", "c"], return_exceptions=True)
    assert results[0] == InputOutputPair(input="a", output="A")
    assert isinstance(results[1], ValueError)
    assert results[2] == InputOutputPair(input="c", output="C")


def test_input_output_runnable_batch_return_exceptions_no_failures() -> None:
    wrapped = InputOutputRunnable(RunnableLambda(lambda x: x.upper()))
    results = wrapped.batch(["a", "b"], return_exceptions=True)
    assert results == [
        InputOutputPair(input="a", output="A"),
        InputOutputPair(input="b", output="B"),
    ]


# --- abatch ---


def test_input_output_runnable_abatch_returns_list_of_input_output_pairs() -> None:
    wrapped = InputOutputRunnable(RunnableLambda(lambda x: x.upper()))
    results = asyncio.run(wrapped.abatch(["a", "b", "c"]))
    assert all(isinstance(r, InputOutputPair) for r in results)


def test_input_output_runnable_abatch_pairs_inputs_and_outputs_in_order() -> None:
    wrapped = InputOutputRunnable(RunnableLambda(lambda x: x.upper()))
    results = asyncio.run(wrapped.abatch(["a", "b", "c"]))
    assert results == [
        InputOutputPair(input="a", output="A"),
        InputOutputPair(input="b", output="B"),
        InputOutputPair(input="c", output="C"),
    ]


def test_input_output_runnable_abatch_uses_inner_abatch_not_inner_ainvoke() -> None:
    inner = _TrackingRunnable()
    wrapped = InputOutputRunnable(inner)
    asyncio.run(wrapped.abatch(["a", "b", "c"]))
    assert inner.abatch_calls == [["a", "b", "c"]]
    assert inner.ainvoke_calls == []


def test_input_output_runnable_abatch_raises_by_default_on_failure() -> None:
    wrapped = InputOutputRunnable(_failing_lambda(fail_on="b"))
    with pytest.raises(ValueError, match="failed for b"):
        asyncio.run(wrapped.abatch(["a", "b", "c"]))


def test_input_output_runnable_abatch_return_exceptions_keeps_raw_exception() -> None:
    wrapped = InputOutputRunnable(_failing_lambda(fail_on="b"))
    results = asyncio.run(wrapped.abatch(["a", "b", "c"], return_exceptions=True))
    assert results[0] == InputOutputPair(input="a", output="A")
    assert isinstance(results[1], ValueError)
    assert results[2] == InputOutputPair(input="c", output="C")


# --- repr ---


def test_input_output_runnable_repr_contains_class_name() -> None:
    wrapped = InputOutputRunnable(RunnableLambda(lambda x: x.upper()))
    assert "InputOutputRunnable" in repr(wrapped)


def test_input_output_runnable_repr_contains_inner_runnable_repr() -> None:
    inner = RunnableLambda(lambda x: x.upper())
    wrapped = InputOutputRunnable(inner)
    assert repr(inner) in repr(wrapped)
