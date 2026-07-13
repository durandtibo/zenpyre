from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from coola.equality import objects_are_equal

from zenpyre.agents.factory import BaseAgentFactory, RecordingAgentFactory

MODULE = "zenpyre.agents.factory.recording"


def _make_factory(**overrides: Any) -> RecordingAgentFactory:
    """Return a RecordingAgentFactory instance for testing."""
    kwargs = {
        "agent_factory": MagicMock(),
        "record_store_factory": MagicMock(),
        "extra": None,
        "serializer": None,
    }
    kwargs.update(overrides)
    return RecordingAgentFactory(**kwargs)


###########################################
#     Tests for RecordingAgentFactory     #
###########################################


# --- Inheritance ---


def test_recording_agent_factory_is_base_agent_factory() -> None:
    assert isinstance(_make_factory(), BaseAgentFactory)


# --- __init__ stores args ---


def test_recording_agent_factory_stores_agent_factory() -> None:
    agent_factory = MagicMock()
    factory = _make_factory(agent_factory=agent_factory)
    assert factory._agent_factory is agent_factory


def test_recording_agent_factory_stores_record_store_factory() -> None:
    record_store_factory = MagicMock()
    factory = _make_factory(record_store_factory=record_store_factory)
    assert factory._record_store_factory is record_store_factory


def test_recording_agent_factory_default_extra_is_none() -> None:
    factory = _make_factory()
    assert factory._extra is None


def test_recording_agent_factory_stores_extra() -> None:
    extra = {"experiment_id": "exp-42"}
    factory = _make_factory(extra=extra)
    assert factory._extra is extra


def test_recording_agent_factory_default_serializer_is_none() -> None:
    factory = _make_factory()
    assert factory._serializer is None


def test_recording_agent_factory_stores_serializer() -> None:
    serializer = lambda x: x  # noqa: E731
    factory = _make_factory(serializer=serializer)
    assert factory._serializer is serializer


# --- make_agent wiring ---


def test_recording_agent_factory_make_agent_builds_agent_from_factory() -> None:
    agent_factory = MagicMock()
    factory = _make_factory(agent_factory=agent_factory)
    with patch(f"{MODULE}.RecordingRunnable"):
        factory.make_agent()
        agent_factory.make_agent.assert_called_once_with()


def test_recording_agent_factory_make_agent_builds_record_store_from_factory() -> None:
    record_store_factory = MagicMock()
    factory = _make_factory(record_store_factory=record_store_factory)
    with patch(f"{MODULE}.RecordingRunnable"):
        factory.make_agent()
        record_store_factory.make_record_store.assert_called_once_with()


def test_recording_agent_factory_make_agent_wraps_in_recording_runnable() -> None:
    agent_factory = MagicMock()
    record_store_factory = MagicMock()
    extra = {"experiment_id": "exp-42"}
    serializer = lambda x: x  # noqa: E731
    factory = _make_factory(
        agent_factory=agent_factory,
        record_store_factory=record_store_factory,
        extra=extra,
        serializer=serializer,
    )
    with patch(f"{MODULE}.RecordingRunnable") as mock_recording_runnable_cls:
        factory.make_agent()
        mock_recording_runnable_cls.assert_called_once_with(
            runnable=agent_factory.make_agent.return_value,
            record_store=record_store_factory.make_record_store.return_value,
            extra=extra,
            serializer=serializer,
        )


def test_recording_agent_factory_make_agent_returns_recording_runnable() -> None:
    factory = _make_factory()
    with patch(f"{MODULE}.RecordingRunnable") as mock_recording_runnable_cls:
        result = factory.make_agent()
        assert result is mock_recording_runnable_cls.return_value


# --- _get_repr_kwargs ---


def test_recording_agent_factory_get_repr_kwargs() -> None:
    agent_factory = MagicMock()
    record_store_factory = MagicMock()
    extra = {"experiment_id": "exp-42"}
    serializer = lambda x: x  # noqa: E731
    factory = _make_factory(
        agent_factory=agent_factory,
        record_store_factory=record_store_factory,
        extra=extra,
        serializer=serializer,
    )
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {
            "agent_factory": agent_factory,
            "record_store_factory": record_store_factory,
            "extra": extra,
            "serializer": serializer,
        },
    )


# --- __repr__ and __str__ ---


def test_recording_agent_factory_repr_starts_with_class_name() -> None:
    factory = _make_factory()
    assert repr(factory).startswith("RecordingAgentFactory(")


def test_recording_agent_factory_str_starts_with_class_name() -> None:
    factory = _make_factory()
    assert str(factory).startswith("RecordingAgentFactory(")


def test_recording_agent_factory_repr_contains_agent_factory() -> None:
    factory = _make_factory()
    assert "agent_factory" in repr(factory)


def test_recording_agent_factory_str_contains_agent_factory() -> None:
    factory = _make_factory()
    assert "agent_factory" in str(factory)
