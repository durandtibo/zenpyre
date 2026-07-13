from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from coola.equality import objects_are_equal
from coola.utils.string import truncate_str
from langchain_core.language_models import BaseChatModel, FakeListChatModel

from zenpyre.agents.factory import AgentChatModelFactory, BaseAgentFactory
from zenpyre.chat_models.factory import BaseChatModelFactory
from zenpyre.utils.config import Config

MODULE = "zenpyre.agents.factory.chat_model"

MINIMAL_CHAT_MODEL_FACTORY_TARGET = (
    "tests.unit.agents.factory.test_chat_model.MinimalChatModelFactory"
)


class MinimalChatModelFactory(BaseChatModelFactory):
    """Minimal concrete BaseChatModelFactory for testing."""

    def make_chat_model(self) -> BaseChatModel:
        return FakeListChatModel(responses=["hello"])


def _make_chat_model_factory() -> MagicMock:
    """Return a MagicMock standing in for a BaseChatModelFactory."""
    return MagicMock(spec=BaseChatModelFactory)


def _make_factory(**overrides: Any) -> AgentChatModelFactory:
    """Return an AgentChatModelFactory instance for testing."""
    kwargs = {
        "chat_model_factory": _make_chat_model_factory(),
        "system_prompt": None,
        "response_format": None,
    }
    kwargs.update(overrides)
    return AgentChatModelFactory(**kwargs)


###########################################
#     Tests for AgentChatModelFactory     #
###########################################


# --- Inheritance ---


def test_agent_chat_model_factory_is_base_agent_factory() -> None:
    assert isinstance(_make_factory(), BaseAgentFactory)


# --- __init__ stores args ---


def test_agent_chat_model_factory_stores_chat_model_factory() -> None:
    chat_model_factory = _make_chat_model_factory()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    assert factory._chat_model_factory is chat_model_factory


def test_agent_chat_model_factory_default_system_prompt_is_none() -> None:
    factory = _make_factory()
    assert factory._system_prompt is None


def test_agent_chat_model_factory_stores_system_prompt() -> None:
    factory = _make_factory(system_prompt="You are helpful.")
    assert factory._system_prompt == "You are helpful."


def test_agent_chat_model_factory_default_response_format_is_none() -> None:
    factory = _make_factory()
    assert factory._response_format is None


def test_agent_chat_model_factory_stores_response_format() -> None:
    response_format = dict
    factory = _make_factory(response_format=response_format)
    assert factory._response_format is response_format


# --- __init__ resolves chat_model_factory ---


def test_agent_chat_model_factory_resolves_chat_model_factory_from_dict() -> None:
    factory = _make_factory(chat_model_factory={"_target_": MINIMAL_CHAT_MODEL_FACTORY_TARGET})
    assert isinstance(factory._chat_model_factory, MinimalChatModelFactory)


def test_agent_chat_model_factory_resolves_chat_model_factory_from_base_config() -> None:
    config = Config.from_kwargs(_target_=MINIMAL_CHAT_MODEL_FACTORY_TARGET)
    factory = _make_factory(chat_model_factory=config)
    assert isinstance(factory._chat_model_factory, MinimalChatModelFactory)


def test_agent_chat_model_factory_invalid_chat_model_factory_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BaseChatModelFactory instance"):
        _make_factory(chat_model_factory="not-a-chat-model-factory")


# --- make_agent wiring ---


def test_agent_chat_model_factory_make_agent_builds_chat_model_from_factory() -> None:
    chat_model_factory = _make_chat_model_factory()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    with patch(f"{MODULE}.AgentChatModel"):
        factory.make_agent()
        chat_model_factory.make_chat_model.assert_called_once_with()


def test_agent_chat_model_factory_make_agent_wraps_in_agent_chat_model() -> None:
    chat_model_factory = _make_chat_model_factory()
    response_format = dict
    factory = _make_factory(
        chat_model_factory=chat_model_factory,
        system_prompt="You are helpful.",
        response_format=response_format,
    )
    with patch(f"{MODULE}.AgentChatModel") as mock_agent_chat_model_cls:
        factory.make_agent()
        mock_agent_chat_model_cls.assert_called_once_with(
            model=chat_model_factory.make_chat_model.return_value,
            system_prompt="You are helpful.",
            response_format=response_format,
        )


def test_agent_chat_model_factory_make_agent_returns_agent_chat_model() -> None:
    factory = _make_factory()
    with patch(f"{MODULE}.AgentChatModel") as mock_agent_chat_model_cls:
        result = factory.make_agent()
        assert result is mock_agent_chat_model_cls.return_value


def test_agent_chat_model_factory_make_agent_builds_new_agent_each_call() -> None:
    chat_model_factory = _make_chat_model_factory()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    with patch(f"{MODULE}.AgentChatModel") as mock_agent_chat_model_cls:
        factory.make_agent()
        factory.make_agent()
        assert mock_agent_chat_model_cls.call_count == 2
        assert chat_model_factory.make_chat_model.call_count == 2


# --- _get_repr_kwargs ---


def test_agent_chat_model_factory_get_repr_kwargs() -> None:
    chat_model_factory = _make_chat_model_factory()
    factory = _make_factory(
        chat_model_factory=chat_model_factory,
        system_prompt="You are helpful.",
        response_format=dict,
    )
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {
            "chat_model_factory": chat_model_factory,
            "system_prompt": "You are helpful.",
            "response_format": dict,
        },
    )


def test_agent_chat_model_factory_get_repr_kwargs_truncates_long_system_prompt() -> None:
    factory = _make_factory(system_prompt="a" * 200)
    assert factory._get_repr_kwargs()["system_prompt"] == truncate_str("a" * 200)
    assert len(factory._get_repr_kwargs()["system_prompt"]) < 200


def test_agent_chat_model_factory_get_repr_kwargs_system_prompt_none() -> None:
    factory = _make_factory(system_prompt=None)
    assert factory._get_repr_kwargs()["system_prompt"] is None


# --- __repr__ and __str__ ---


def test_agent_chat_model_factory_repr_starts_with_class_name() -> None:
    factory = _make_factory()
    assert repr(factory).startswith("AgentChatModelFactory(")


def test_agent_chat_model_factory_str_starts_with_class_name() -> None:
    factory = _make_factory()
    assert str(factory).startswith("AgentChatModelFactory(")


def test_agent_chat_model_factory_repr_contains_chat_model_factory() -> None:
    factory = _make_factory()
    assert "chat_model_factory" in repr(factory)


def test_agent_chat_model_factory_str_contains_chat_model_factory() -> None:
    factory = _make_factory()
    assert "chat_model_factory" in str(factory)
