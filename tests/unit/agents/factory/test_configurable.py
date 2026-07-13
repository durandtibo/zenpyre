from __future__ import annotations

from coola.equality import objects_are_equal
from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import Runnable

from zenpyre.agents import AgentChatModel
from zenpyre.agents.factory import (
    BaseAgentFactory,
    ConfigurableAgentFactory,
)

AGENT_CHAT_MODEL_TARGET = "zenpyre.agents.AgentChatModel"


def _make_agent() -> AgentChatModel:
    """Return an AgentChatModel instance for testing."""
    return AgentChatModel(model=FakeListChatModel(responses=["hello"]))


##############################################
#     Tests for ConfigurableAgentFactory     #
##############################################


# --- Inheritance ---


def test_configurable_agent_factory_is_base_agent_factory() -> None:
    assert isinstance(ConfigurableAgentFactory(_make_agent()), BaseAgentFactory)


# --- make_agent from instance ---


def test_configurable_agent_factory_make_agent_returns_runnable() -> None:
    factory = ConfigurableAgentFactory(_make_agent())
    assert isinstance(factory.make_agent(), Runnable)


def test_configurable_agent_factory_make_agent_returns_same_instance() -> None:
    agent = _make_agent()
    factory = ConfigurableAgentFactory(agent)
    assert factory.make_agent() is agent


# --- make_agent from dict ---


def test_configurable_agent_factory_make_agent_from_dict_returns_runnable() -> None:
    factory = ConfigurableAgentFactory(
        {"_target_": AGENT_CHAT_MODEL_TARGET, "model": FakeListChatModel(responses=["hello"])}
    )
    assert isinstance(factory.make_agent(), Runnable)


def test_configurable_agent_factory_make_agent_from_dict_returns_correct_type() -> None:
    factory = ConfigurableAgentFactory(
        {"_target_": AGENT_CHAT_MODEL_TARGET, "model": FakeListChatModel(responses=["hello"])}
    )
    assert isinstance(factory.make_agent(), AgentChatModel)


# --- _get_repr_kwargs ---


def test_configurable_agent_factory_get_repr_kwargs_instance() -> None:
    agent = _make_agent()
    factory = ConfigurableAgentFactory(agent)
    assert objects_are_equal(factory._get_repr_kwargs(), {"agent": agent})


def test_configurable_agent_factory_get_repr_kwargs_dict_input() -> None:
    config = {"_target_": AGENT_CHAT_MODEL_TARGET, "model": FakeListChatModel(responses=["hello"])}
    factory = ConfigurableAgentFactory(config)
    assert objects_are_equal(factory._get_repr_kwargs(), {"agent": config})


# --- __repr__ and __str__ ---


def test_configurable_agent_factory_repr_starts_with_class_name() -> None:
    factory = ConfigurableAgentFactory(_make_agent())
    assert repr(factory).startswith("ConfigurableAgentFactory(")


def test_configurable_agent_factory_str_starts_with_class_name() -> None:
    factory = ConfigurableAgentFactory(_make_agent())
    assert str(factory).startswith("ConfigurableAgentFactory(")


def test_configurable_agent_factory_repr_contains_agent() -> None:
    factory = ConfigurableAgentFactory(_make_agent())
    assert "agent" in repr(factory)


def test_configurable_agent_factory_str_contains_agent() -> None:
    factory = ConfigurableAgentFactory(_make_agent())
    assert "agent" in str(factory)
