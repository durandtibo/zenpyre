from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from coola.equality import objects_are_equal

from zenpyre.agents.factory import BaseAgentFactory, CreateAgentFactory

MODULE = "zenpyre.agents.factory.create_agent"


def _make_factory(**overrides: Any) -> CreateAgentFactory:
    """Return a CreateAgentFactory instance for testing."""
    kwargs = {"chat_model_factory": MagicMock()}
    kwargs.update(overrides)
    return CreateAgentFactory(**kwargs)


########################################
#     Tests for CreateAgentFactory     #
########################################


# --- Inheritance ---


def test_create_agent_factory_is_base_agent_factory() -> None:
    assert isinstance(_make_factory(), BaseAgentFactory)


# --- __init__ stores args ---


def test_create_agent_factory_stores_chat_model_factory() -> None:
    chat_model_factory = MagicMock()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    assert factory._chat_model_factory is chat_model_factory


def test_create_agent_factory_default_kwargs_is_empty() -> None:
    factory = _make_factory()
    assert factory._kwargs == {}


def test_create_agent_factory_stores_kwargs() -> None:
    tools = [MagicMock()]
    factory = _make_factory(tools=tools, system_prompt="You are helpful.", name="my-agent")
    assert factory._kwargs == {
        "tools": tools,
        "system_prompt": "You are helpful.",
        "name": "my-agent",
    }


# --- make_agent wiring ---


def test_create_agent_factory_make_agent_builds_chat_model_from_factory() -> None:
    chat_model_factory = MagicMock()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    with patch(f"{MODULE}.create_agent"):
        factory.make_agent()
        chat_model_factory.make_chat_model.assert_called_once_with()


def test_create_agent_factory_make_agent_calls_create_agent() -> None:
    chat_model_factory = MagicMock()
    tools = [MagicMock()]
    response_format = dict
    factory = _make_factory(
        chat_model_factory=chat_model_factory,
        tools=tools,
        system_prompt="You are helpful.",
        response_format=response_format,
        name="my-agent",
    )
    with patch(f"{MODULE}.create_agent") as mock_create_agent:
        factory.make_agent()
        mock_create_agent.assert_called_once_with(
            model=chat_model_factory.make_chat_model.return_value,
            tools=tools,
            system_prompt="You are helpful.",
            response_format=response_format,
            name="my-agent",
        )


def test_create_agent_factory_make_agent_calls_create_agent_with_no_extra_kwargs() -> None:
    chat_model_factory = MagicMock()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    with patch(f"{MODULE}.create_agent") as mock_create_agent:
        factory.make_agent()
        mock_create_agent.assert_called_once_with(
            model=chat_model_factory.make_chat_model.return_value
        )


def test_create_agent_factory_make_agent_returns_create_agent_result() -> None:
    factory = _make_factory()
    with patch(f"{MODULE}.create_agent") as mock_create_agent:
        result = factory.make_agent()
        assert result is mock_create_agent.return_value


def test_create_agent_factory_make_agent_builds_new_agent_each_call() -> None:
    chat_model_factory = MagicMock()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    with patch(f"{MODULE}.create_agent") as mock_create_agent:
        factory.make_agent()
        factory.make_agent()
        assert mock_create_agent.call_count == 2
        assert chat_model_factory.make_chat_model.call_count == 2


# --- _get_repr_kwargs ---


def test_create_agent_factory_get_repr_kwargs() -> None:
    chat_model_factory = MagicMock()
    tools = [MagicMock()]
    factory = _make_factory(
        chat_model_factory=chat_model_factory,
        tools=tools,
        system_prompt="You are helpful.",
        response_format=dict,
        name="my-agent",
    )
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {
            "chat_model_factory": chat_model_factory,
            "tools": tools,
            "system_prompt": "You are helpful.",
            "response_format": dict,
            "name": "my-agent",
        },
    )


def test_create_agent_factory_get_repr_kwargs_with_no_extra_kwargs() -> None:
    chat_model_factory = MagicMock()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    assert objects_are_equal(factory._get_repr_kwargs(), {"chat_model_factory": chat_model_factory})


# --- __repr__ and __str__ ---


def test_create_agent_factory_repr_starts_with_class_name() -> None:
    factory = _make_factory()
    assert repr(factory).startswith("CreateAgentFactory(")


def test_create_agent_factory_str_starts_with_class_name() -> None:
    factory = _make_factory()
    assert str(factory).startswith("CreateAgentFactory(")


def test_create_agent_factory_repr_contains_chat_model_factory() -> None:
    factory = _make_factory()
    assert "chat_model_factory" in repr(factory)


def test_create_agent_factory_str_contains_chat_model_factory() -> None:
    factory = _make_factory()
    assert "chat_model_factory" in str(factory)
