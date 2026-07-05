from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import pytest

from zenpyre.agents import AgentConfig


@dataclass(frozen=True)
class OpenAIAgentConfig(AgentConfig):
    max_tokens: int | None = None


def make_config(**kwargs: Any) -> AgentConfig:
    defaults = {"model": "openai:gpt-4o", "system_prompt": "You are helpful."}
    defaults.update(kwargs)
    return AgentConfig(**defaults)


#################################
#     Tests for AgentConfig     #
#################################

# --- Constructor ---


def test_agent_config_stores_model() -> None:
    assert make_config(model="openai:gpt-4o").model == "openai:gpt-4o"


def test_agent_config_stores_system_prompt() -> None:
    assert make_config(system_prompt="You are helpful.").system_prompt == "You are helpful."


def test_agent_config_temperature_default() -> None:
    assert make_config().temperature == 0.0


def test_agent_config_stores_temperature() -> None:
    assert make_config(temperature=0.7).temperature == 0.7


# --- Immutability ---


def test_agent_config_is_frozen() -> None:
    config = make_config()
    with pytest.raises(FrozenInstanceError):
        config.model = "other"


def test_agent_config_is_hashable() -> None:
    config = make_config()
    assert isinstance(hash(config), int)


def test_agent_config_equal_configs_are_equal() -> None:
    assert make_config() == make_config()


def test_agent_config_different_configs_are_not_equal() -> None:
    assert make_config(model="openai:gpt-4o") != make_config(model="openai:gpt-4o-mini")


# --- repr ---


def test_agent_config_repr() -> None:
    assert repr(make_config()).startswith("AgentConfig(")


# --- cache_key: return type ---


def test_agent_config_cache_key_returns_str() -> None:
    assert isinstance(make_config().cache_key(), str)


def test_agent_config_cache_key_default_length() -> None:
    assert len(make_config().cache_key()) == 64


# --- cache_key: stability ---


def test_agent_config_cache_key_same_config_same_hash() -> None:
    config = make_config()
    assert config.cache_key() == config.cache_key()


def test_agent_config_cache_key_equal_configs_same_hash() -> None:
    assert make_config().cache_key() == make_config().cache_key()


def test_agent_config_cache_key_different_model_different_hash() -> None:
    config_a = make_config(model="openai:gpt-4o")
    config_b = make_config(model="openai:gpt-4o-mini")
    assert config_a.cache_key() != config_b.cache_key()


def test_agent_config_cache_key_different_system_prompt_different_hash() -> None:
    config_a = make_config(system_prompt="Prompt A")
    config_b = make_config(system_prompt="Prompt B")
    assert config_a.cache_key() != config_b.cache_key()


def test_agent_config_cache_key_different_temperature_different_hash() -> None:
    config_a = make_config(temperature=0.0)
    config_b = make_config(temperature=0.7)
    assert config_a.cache_key() != config_b.cache_key()


# --- Subclassing ---


def test_openai_agent_config_is_agent_config_instance() -> None:
    config = OpenAIAgentConfig(model="openai:gpt-4o", system_prompt="Hi", max_tokens=1024)
    assert isinstance(config, AgentConfig)


def test_openai_agent_config_inherits_base_fields() -> None:
    config = OpenAIAgentConfig(model="openai:gpt-4o", system_prompt="Hi", temperature=0.5)
    assert config.model == "openai:gpt-4o"
    assert config.system_prompt == "Hi"
    assert config.temperature == 0.5


def test_openai_agent_config_max_tokens_default() -> None:
    config = OpenAIAgentConfig(model="openai:gpt-4o", system_prompt="Hi")
    assert config.max_tokens is None


def test_openai_agent_config_stores_max_tokens() -> None:
    config = OpenAIAgentConfig(model="openai:gpt-4o", system_prompt="Hi", max_tokens=1024)
    assert config.max_tokens == 1024


def test_openai_agent_config_is_frozen() -> None:
    config = OpenAIAgentConfig(model="openai:gpt-4o", system_prompt="Hi")
    with pytest.raises(FrozenInstanceError):
        config.max_tokens = 2048


# --- Subclass cache_key ---


def test_openai_agent_config_cache_key_returns_str() -> None:
    config = OpenAIAgentConfig(model="openai:gpt-4o", system_prompt="Hi", max_tokens=1024)
    assert isinstance(config.cache_key(), str)


def test_openai_agent_config_cache_key_includes_subclass_field() -> None:
    config_a = OpenAIAgentConfig(model="openai:gpt-4o", system_prompt="Hi", max_tokens=1024)
    config_b = OpenAIAgentConfig(model="openai:gpt-4o", system_prompt="Hi", max_tokens=2048)
    assert config_a.cache_key() != config_b.cache_key()


def test_openai_agent_config_cache_key_differs_from_base_with_same_common_fields() -> None:
    """A subclass instance and a base instance sharing the same common
    field values should still hash differently, since the subclass
    carries an additional field (max_tokens) that changes what gets
    hashed."""
    base_config = AgentConfig(model="openai:gpt-4o", system_prompt="Hi")
    sub_config = OpenAIAgentConfig(model="openai:gpt-4o", system_prompt="Hi", max_tokens=None)
    assert base_config.cache_key() != sub_config.cache_key()
