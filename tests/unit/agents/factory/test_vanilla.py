from __future__ import annotations

from coola.equality import objects_are_equal
from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import Runnable

from zenpyre.agents import AgentChatModel
from zenpyre.agents.factory import AgentFactory, BaseAgentFactory


def _make_agent() -> AgentChatModel:
    """Return an AgentChatModel instance for testing."""
    return AgentChatModel(model=FakeListChatModel(responses=["hello"]))


##############################################
#     Tests for AgentFactory                 #
##############################################


# --- Inheritance ---


def test_agent_factory_is_base_agent_factory() -> None:
    assert isinstance(AgentFactory(_make_agent()), BaseAgentFactory)


# --- make_agent ---


def test_agent_factory_make_agent_returns_runnable() -> None:
    factory = AgentFactory(_make_agent())
    assert isinstance(factory.make_agent(), Runnable)


def test_agent_factory_make_agent_returns_same_instance() -> None:
    agent = _make_agent()
    factory = AgentFactory(agent)
    assert factory.make_agent() is agent


def test_agent_factory_make_agent_returns_same_instance_across_calls() -> None:
    agent = _make_agent()
    factory = AgentFactory(agent)
    assert factory.make_agent() is factory.make_agent()


# --- _get_repr_kwargs ---


def test_agent_factory_get_repr_kwargs() -> None:
    agent = _make_agent()
    factory = AgentFactory(agent)
    assert objects_are_equal(factory._get_repr_kwargs(), {"agent": agent})


# --- __repr__ and __str__ ---


def test_agent_factory_repr_starts_with_class_name() -> None:
    factory = AgentFactory(_make_agent())
    assert repr(factory).startswith("AgentFactory(")


def test_agent_factory_str_starts_with_class_name() -> None:
    factory = AgentFactory(_make_agent())
    assert str(factory).startswith("AgentFactory(")


def test_agent_factory_repr_contains_agent() -> None:
    factory = AgentFactory(_make_agent())
    assert "agent" in repr(factory)


def test_agent_factory_str_contains_agent() -> None:
    factory = AgentFactory(_make_agent())
    assert "agent" in str(factory)
