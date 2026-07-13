from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from coola.equality import objects_are_equal
from coola.utils.string import truncate_str
from langchain_core.language_models import FakeListChatModel

from zenpyre.agents.factory import BaseAgentFactory, CreateAgentFactory
from zenpyre.chat_models.factory import BaseChatModelFactory
from zenpyre.utils.config import Config

MODULE = "zenpyre.agents.factory.create_agent"

MINIMAL_CHAT_MODEL_FACTORY_TARGET = (
    "tests.unit.agents.factory.test_create_agent.MinimalChatModelFactory"
)


class MinimalChatModelFactory(BaseChatModelFactory):
    """Minimal concrete BaseChatModelFactory for testing."""

    def make_chat_model(self) -> FakeListChatModel:
        return FakeListChatModel(responses=["hello"])


def _make_chat_model_factory() -> MagicMock:
    """Return a MagicMock standing in for a BaseChatModelFactory."""
    return MagicMock(spec=BaseChatModelFactory)


def _make_factory(**overrides: Any) -> CreateAgentFactory:
    """Return a CreateAgentFactory instance for testing."""
    kwargs = {"chat_model_factory": _make_chat_model_factory()}
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
    chat_model_factory = _make_chat_model_factory()
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


# --- __init__ resolves chat_model_factory ---


def test_create_agent_factory_resolves_chat_model_factory_from_dict() -> None:
    factory = _make_factory(chat_model_factory={"_target_": MINIMAL_CHAT_MODEL_FACTORY_TARGET})
    assert isinstance(factory._chat_model_factory, MinimalChatModelFactory)


def test_create_agent_factory_resolves_chat_model_factory_from_base_config() -> None:
    config = Config.from_kwargs(_target_=MINIMAL_CHAT_MODEL_FACTORY_TARGET)
    factory = _make_factory(chat_model_factory=config)
    assert isinstance(factory._chat_model_factory, MinimalChatModelFactory)


def test_create_agent_factory_invalid_chat_model_factory_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BaseChatModelFactory instance"):
        _make_factory(chat_model_factory="not-a-chat-model-factory")


# --- make_agent wiring ---


def test_create_agent_factory_make_agent_builds_chat_model_from_factory() -> None:
    chat_model_factory = _make_chat_model_factory()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    with patch(f"{MODULE}.create_agent"):
        factory.make_agent()
        chat_model_factory.make_chat_model.assert_called_once_with()


def test_create_agent_factory_make_agent_calls_create_agent() -> None:
    chat_model_factory = _make_chat_model_factory()
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
    chat_model_factory = _make_chat_model_factory()
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
    chat_model_factory = _make_chat_model_factory()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    with patch(f"{MODULE}.create_agent") as mock_create_agent:
        factory.make_agent()
        factory.make_agent()
        assert mock_create_agent.call_count == 2
        assert chat_model_factory.make_chat_model.call_count == 2


# --- _get_repr_kwargs ---


def test_create_agent_factory_get_repr_kwargs() -> None:
    chat_model_factory = _make_chat_model_factory()
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
    chat_model_factory = _make_chat_model_factory()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    assert objects_are_equal(factory._get_repr_kwargs(), {"chat_model_factory": chat_model_factory})


def test_create_agent_factory_get_repr_kwargs_truncates_long_system_prompt() -> None:
    factory = _make_factory(system_prompt="a" * 200)
    assert factory._get_repr_kwargs()["system_prompt"] == truncate_str("a" * 200)
    assert len(factory._get_repr_kwargs()["system_prompt"]) < 200


def test_create_agent_factory_get_repr_kwargs_no_system_prompt() -> None:
    factory = _make_factory()
    assert "system_prompt" not in factory._get_repr_kwargs()


def test_create_agent_factory_get_repr_kwargs_non_str_system_prompt_left_as_is() -> None:
    system_prompt = MagicMock()
    factory = _make_factory(system_prompt=system_prompt)
    assert factory._get_repr_kwargs()["system_prompt"] is system_prompt


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
