from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.runnables import RunnableGenerator, RunnableLambda
from pydantic import BaseModel

from zenpyre.record_stores import InMemoryRecordStore
from zenpyre.runnables import RecordingRunnable
from zenpyre.runnables.recording import _try_add

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryRecordStore:
    return InMemoryRecordStore()


@pytest.fixture
def upper_runnable() -> RunnableLambda:
    return RunnableLambda(lambda x: x.upper())


@pytest.fixture
def failing_runnable() -> RunnableLambda:
    def _call(x: str) -> str:
        if x == "bad":
            msg = "boom"
            raise ValueError(msg)
        return x.upper()

    return RunnableLambda(_call)


def _string_chunks(
    input: str,  # noqa: ARG001, A002
    config: dict | None = None,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> Iterator[str]:
    yield from ["Hel", "lo"]


def _no_chunks(
    input: str,  # noqa: ARG001, A002
    config: dict | None = None,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> Iterator[str]:
    return
    yield  # pragma: no cover -- makes this a generator function


async def _async_string_chunks(
    input: str,  # noqa: ARG001, A002
    config: dict | None = None,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> AsyncIterator[str]:
    for chunk in ["Hel", "lo"]:
        yield chunk


async def _no_async_chunks(
    input: str,  # noqa: ARG001, A002
    config: dict | None = None,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> AsyncIterator[str]:
    return
    yield  # pragma: no cover -- makes this an async generator function


#######################################
#     Tests for RecordingRunnable     #
#######################################


def test_init_basic(upper_runnable: RunnableLambda, store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    assert recorded.invoke("hi") == "HI"


def test_init_extra_reserved_key_raises(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    with pytest.raises(ValueError, match=r"'extra' must not contain a reserved key"):
        RecordingRunnable(upper_runnable, store, extra={"input": "x"})


def test_init_extra_is_not_aliased(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    original = {"experiment_id": "exp-A"}
    recorded = RecordingRunnable(upper_runnable, store, extra=original)
    original["experiment_id"] = "MUTATED"
    recorded.invoke("hi")
    assert store.all()[0].metadata["experiment_id"] == "exp-A"


# --- invoke ---


def test_invoke_returns_output(upper_runnable: RunnableLambda, store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    assert recorded.invoke("hello") == "HELLO"


def test_invoke_writes_one_record(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    recorded.invoke("hello")
    assert store.count() == 1


def test_invoke_records_input_and_output(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    recorded.invoke("hello")
    metadata = store.all()[0].metadata
    assert metadata["input"] == "hello"
    assert metadata["output"] == "HELLO"


def test_invoke_records_no_error(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    recorded.invoke("hello")
    assert store.all()[0].metadata["error"] is None


def test_invoke_records_run_id(upper_runnable: RunnableLambda, store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    recorded.invoke("hi", config={"run_id": "abc-123"})
    assert store.all()[0].metadata["run_id"] == "abc-123"


def test_invoke_run_id_defaults_to_none(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    recorded.invoke("hi")
    assert store.all()[0].metadata["run_id"] is None


def test_invoke_includes_extra(upper_runnable: RunnableLambda, store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(upper_runnable, store, extra={"experiment_id": "exp-42"})
    recorded.invoke("hi")
    assert store.all()[0].metadata["experiment_id"] == "exp-42"


def test_invoke_includes_config_metadata(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    recorded.invoke("hi", config={"metadata": {"session_id": "s-1"}})
    assert store.all()[0].metadata["session_id"] == "s-1"


def test_invoke_config_metadata_overrides_extra(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store, extra={"experiment_id": "exp-42"})
    recorded.invoke("hi", config={"metadata": {"experiment_id": "exp-999"}})
    assert store.all()[0].metadata["experiment_id"] == "exp-999"


def test_invoke_config_metadata_reserved_key_raises(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    with pytest.raises(ValueError, match=r"config\['metadata'\] must not contain a reserved key"):
        recorded.invoke("hi", config={"metadata": {"output": "sneaky"}})


def test_invoke_failure_propagates(
    failing_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(failing_runnable, store)
    with pytest.raises(ValueError, match="boom"):
        recorded.invoke("bad")


def test_invoke_failure_writes_no_record(
    failing_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(failing_runnable, store)
    with pytest.raises(ValueError, match="boom"):
        recorded.invoke("bad")
    assert store.count() == 0


def test_invoke_two_identical_calls_produce_two_records(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    recorded.invoke("same")
    recorded.invoke("same")
    assert store.count() == 2


def test_invoke_two_identical_calls_produce_distinct_ids(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    recorded.invoke("same")
    recorded.invoke("same")
    ids = [r.id for r in store.all()]
    assert len(set(ids)) == 2


# --- ainvoke ---


def test_ainvoke_returns_output(upper_runnable: RunnableLambda, store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    assert asyncio.run(recorded.ainvoke("hello")) == "HELLO"


def test_ainvoke_writes_one_record(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    asyncio.run(recorded.ainvoke("hello"))
    assert store.count() == 1


def test_ainvoke_failure_writes_no_record(
    failing_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(failing_runnable, store)
    with pytest.raises(ValueError, match="boom"):
        asyncio.run(recorded.ainvoke("bad"))
    assert store.count() == 0


# --- batch ---


def test_batch_returns_results(upper_runnable: RunnableLambda, store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    assert recorded.batch(["a", "b", "c"]) == ["A", "B", "C"]


def test_batch_writes_one_record_per_item(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    recorded.batch(["a", "b", "c"])
    assert store.count() == 3


def test_batch_records_match_inputs(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    recorded.batch(["a", "b"])
    inputs = {r.metadata["input"] for r in store.all()}
    assert inputs == {"a", "b"}


def test_batch_with_return_exceptions_records_error(
    failing_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(failing_runnable, store)
    recorded.batch(["ok", "bad"], return_exceptions=True)
    bad_record = next(r for r in store.all() if r.metadata["input"] == "bad")
    assert bad_record.metadata["output"] is None
    assert "boom" in bad_record.metadata["error"]


def test_batch_with_return_exceptions_still_records_success(
    failing_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(failing_runnable, store)
    recorded.batch(["ok", "bad"], return_exceptions=True)
    ok_record = next(r for r in store.all() if r.metadata["input"] == "ok")
    assert ok_record.metadata["output"] == "OK"
    assert ok_record.metadata["error"] is None


def test_batch_without_return_exceptions_propagates(
    failing_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(failing_runnable, store)
    with pytest.raises(ValueError, match="boom"):
        recorded.batch(["ok", "bad"])


def test_batch_per_item_config_metadata(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    recorded.batch(
        ["a", "b"],
        config=[{"metadata": {"session_id": "s-a"}}, {"metadata": {"session_id": "s-b"}}],
    )
    session_ids = {r.metadata["input"]: r.metadata["session_id"] for r in store.all()}
    assert session_ids == {"a": "s-a", "b": "s-b"}


# --- abatch ---


def test_abatch_returns_results(upper_runnable: RunnableLambda, store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    assert asyncio.run(recorded.abatch(["a", "b"])) == ["A", "B"]


def test_abatch_writes_one_record_per_item(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    asyncio.run(recorded.abatch(["a", "b"]))
    assert store.count() == 2


# --- stream ---


def test_stream_yields_chunks_unchanged(store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(RunnableGenerator(_string_chunks), store)
    assert list(recorded.stream("go")) == ["Hel", "lo"]


def test_stream_records_accumulated_output(store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(RunnableGenerator(_string_chunks), store)
    list(recorded.stream("go"))
    assert store.all()[0].metadata["output"] == "Hello"


def test_stream_records_original_input(store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(RunnableGenerator(_string_chunks), store)
    list(recorded.stream("go"))
    assert store.all()[0].metadata["input"] == "go"


def test_stream_writes_exactly_one_record(store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(RunnableGenerator(_string_chunks), store)
    list(recorded.stream("go"))
    assert store.count() == 1


def test_stream_with_no_chunks_writes_no_record(store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(RunnableGenerator(_no_chunks), store)
    list(recorded.stream("go"))
    assert store.count() == 0


def test_stream_with_no_chunks_logs_warning(
    store: InMemoryRecordStore, caplog: pytest.LogCaptureFixture
) -> None:
    recorded = RecordingRunnable(RunnableGenerator(_no_chunks), store)
    with caplog.at_level("WARNING"):
        list(recorded.stream("go"))
    assert "no record was written for this call" in caplog.text


# --- astream ---


def test_astream_yields_chunks_unchanged(store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(RunnableGenerator(_async_string_chunks), store)

    async def _collect() -> list[str]:
        return [chunk async for chunk in recorded.astream("go")]

    assert asyncio.run(_collect()) == ["Hel", "lo"]


def test_astream_records_accumulated_output(store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(RunnableGenerator(_async_string_chunks), store)

    async def _consume() -> None:
        async for _ in recorded.astream("go"):
            pass

    asyncio.run(_consume())
    assert store.all()[0].metadata["output"] == "Hello"


def test_astream_with_no_chunks_writes_no_record(store: InMemoryRecordStore) -> None:
    recorded = RecordingRunnable(RunnableGenerator(_no_async_chunks), store)

    async def _consume() -> None:
        async for _ in recorded.astream("go"):
            pass

    asyncio.run(_consume())
    assert store.count() == 0


def test_astream_with_no_chunks_logs_warning(
    store: InMemoryRecordStore, caplog: pytest.LogCaptureFixture
) -> None:
    recorded = RecordingRunnable(RunnableGenerator(_no_async_chunks), store)

    async def _consume() -> None:
        async for _ in recorded.astream("go"):
            pass

    with caplog.at_level("WARNING"):
        asyncio.run(_consume())
    assert "no record was written for this call" in caplog.text


# --- reserved_metadata_keys ---


def test_reserved_metadata_keys_default(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    assert recorded.reserved_metadata_keys == {"input", "output", "timestamp", "run_id", "error"}


def test_reserved_metadata_keys_subclass_override(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    class CustomRecordingRunnable(RecordingRunnable):
        @property
        def reserved_metadata_keys(self) -> frozenset[str]:
            return frozenset({"input", "output", "timestamp", "run_id", "error", "secret"})

    recorded = CustomRecordingRunnable(upper_runnable, store)
    with pytest.raises(ValueError, match=r"secret"):
        recorded.invoke("x", config={"metadata": {"secret": "leak"}})


# --- serializer ---


def test_default_serializer_applied(store: InMemoryRecordStore) -> None:
    class Answer(BaseModel):
        value: int

    recorded = RecordingRunnable(RunnableLambda(lambda x: Answer(value=len(x))), store)
    recorded.invoke("hello")
    assert store.all()[0].metadata["output"] == {"value": 5}


def test_custom_serializer_applied_to_whole_metadata(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    def redact(metadata: dict) -> dict:
        metadata = dict(metadata)
        metadata["input"] = "<redacted>"
        return metadata

    recorded = RecordingRunnable(upper_runnable, store, serializer=redact)
    recorded.invoke("secret")
    assert store.all()[0].metadata["input"] == "<redacted>"


# --- repr/str ---


def test_repr_contains_class_name(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    assert repr(recorded).startswith("RecordingRunnable(")


def test_str_contains_class_name(
    upper_runnable: RunnableLambda, store: InMemoryRecordStore
) -> None:
    recorded = RecordingRunnable(upper_runnable, store)
    assert str(recorded).startswith("RecordingRunnable(")


##############################
#     Tests for _try_add     #
##############################


def test_try_add_strings() -> None:
    assert _try_add("Hel", "lo") == "Hello"


def test_try_add_ints() -> None:
    assert _try_add(1, 2) == 3


def test_try_add_lists() -> None:
    assert _try_add([1, 2], [3]) == [1, 2, 3]


def test_try_add_ai_message_chunks() -> None:
    from langchain_core.messages import AIMessageChunk

    result = _try_add(AIMessageChunk(content="Hel"), AIMessageChunk(content="lo"))
    assert result.content == "Hello"


def test_try_add_incompatible_types_returns_accumulated_unchanged() -> None:
    assert _try_add(1, "not an int") == 1


def test_try_add_incompatible_types_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        _try_add(1, "not an int")
    assert "does not support '+'" in caplog.text


def test_try_add_warning_message_includes_chunk_type_name(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        _try_add(1, "not an int")
    assert "str" in caplog.text
