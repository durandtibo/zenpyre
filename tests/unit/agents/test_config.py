from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from types import MappingProxyType

import pytest
from coola.hashing import hash_string

from zenpyre.agents import AgentConfig
from zenpyre.chat_models import ChatModelConfig

# ---------------------------------------------------------------------------
# Fixtures / test doubles
# ---------------------------------------------------------------------------


@pytest.fixture
def chat_model() -> ChatModelConfig:
    return ChatModelConfig(model="gpt-4")


@dataclass(frozen=True)
class MyAgentConfig(AgentConfig):
    """A minimal subclass adding one typed field, mirroring the pattern
    shown in AgentConfig's own class docstring."""

    max_tokens: int | None = None

    def __hash__(self) -> int:
        return AgentConfig.__hash__(self)


#################################
#     Tests for AgentConfig     #
#################################


def test_agent_config_basic_construction(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="You are helpful.", system_prompt_id="id-1"
    )
    assert cfg.chat_model is chat_model
    assert cfg.system_prompt == "You are helpful."
    assert cfg.system_prompt_id == "id-1"
    assert dict(cfg.extra) == {}


def test_agent_config_with_extra(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"max_retries": 3}
    )
    assert dict(cfg.extra) == {"max_retries": 3}


def test_agent_config_extra_stored_as_mapping_proxy(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"max_retries": 3}
    )
    assert isinstance(cfg.extra, MappingProxyType)


def test_agent_config_extra_dict_not_shared_with_caller(chat_model: ChatModelConfig) -> None:
    original = {"max_retries": 3}
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra=original
    )
    original["max_retries"] = 999
    assert cfg.extra["max_retries"] == 3


def test_agent_config_is_frozen(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-1")
    with pytest.raises(FrozenInstanceError):
        cfg.system_prompt = "other"


def test_agent_config_frozen_applies_to_chat_model_and_extra_too(
    chat_model: ChatModelConfig,
) -> None:
    cfg = AgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-1")
    with pytest.raises(FrozenInstanceError):
        cfg.chat_model = ChatModelConfig(model="gpt-3.5")
    with pytest.raises(FrozenInstanceError):
        cfg.extra = {}


def test_agent_config_extra_mutation_after_construction_raises(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"max_retries": 3}
    )
    with pytest.raises(TypeError):
        cfg.extra["max_retries"] = 999


def test_extra_with_chat_model_key_raises(chat_model: ChatModelConfig) -> None:
    with pytest.raises(
        ValueError, match=r"'extra' must not contain any of this config's own field names"
    ):
        AgentConfig(
            chat_model=chat_model,
            system_prompt="p",
            system_prompt_id="id-1",
            extra={"chat_model": "x"},
        )


def test_extra_with_system_prompt_key_raises(chat_model: ChatModelConfig) -> None:
    with pytest.raises(
        ValueError, match=r"'extra' must not contain any of this config's own field names"
    ):
        AgentConfig(
            chat_model=chat_model,
            system_prompt="p",
            system_prompt_id="id-1",
            extra={"system_prompt": "x"},
        )


def test_extra_with_system_prompt_id_key_raises(chat_model: ChatModelConfig) -> None:
    with pytest.raises(
        ValueError, match=r"'extra' must not contain any of this config's own field names"
    ):
        AgentConfig(
            chat_model=chat_model,
            system_prompt="p",
            system_prompt_id="id-1",
            extra={"system_prompt_id": "x"},
        )


def test_extra_with_unrelated_key_does_not_raise(chat_model: ChatModelConfig) -> None:
    AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"max_retries": 3}
    )  # should not raise


def test_extra_collision_error_lists_reserved_names(chat_model: ChatModelConfig) -> None:
    with pytest.raises(ValueError, match=r"chat_model") as exc_info:
        AgentConfig(
            chat_model=chat_model,
            system_prompt="p",
            system_prompt_id="id-1",
            extra={"chat_model": "x"},
        )
    assert "system_prompt" in str(exc_info.value)


def test_subclass_field_collision_also_raises(chat_model: ChatModelConfig) -> None:
    """The collision guard must protect subclass-added fields too, not
    just the base class's own fields (this is the bug the original,
    hardcoded-"model"-only check would have missed)."""
    with pytest.raises(ValueError, match=r"max_tokens"):
        MyAgentConfig(
            chat_model=chat_model,
            system_prompt="p",
            system_prompt_id="id-1",
            extra={"max_tokens": 999},
        )


# --- to_kwargs ---


def test_to_kwargs_contains_chat_model_and_system_prompt(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="You are helpful.", system_prompt_id="id-1"
    )
    kwargs = cfg.to_kwargs()
    assert kwargs["chat_model"] is chat_model
    assert kwargs["system_prompt"] == "You are helpful."


def test_to_kwargs_contains_system_prompt_id(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-1")
    assert cfg.to_kwargs()["system_prompt_id"] == "id-1"


def test_to_kwargs_merges_extra(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"max_retries": 3}
    )
    kwargs = cfg.to_kwargs()
    assert kwargs["max_retries"] == 3


def test_to_kwargs_does_not_include_extra_key_itself(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"max_retries": 3}
    )
    assert "extra" not in cfg.to_kwargs()


def test_to_kwargs_returns_plain_dict(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-1")
    assert type(cfg.to_kwargs()) is dict


def test_to_kwargs_mutating_result_does_not_affect_config(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"max_retries": 3}
    )
    kwargs = cfg.to_kwargs()
    kwargs["max_retries"] = 999
    assert cfg.extra["max_retries"] == 3


def test_to_kwargs_subclass_field_is_automatically_included(chat_model: ChatModelConfig) -> None:
    """This is the key behavioral fix: a subclass-added field must
    appear in to_kwargs() without the subclass overriding anything."""
    cfg = MyAgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", max_tokens=1024
    )
    kwargs = cfg.to_kwargs()
    assert kwargs["max_tokens"] == 1024
    assert kwargs["chat_model"] is chat_model
    assert kwargs["system_prompt"] == "p"


def test_to_kwargs_subclass_field_alongside_extra(chat_model: ChatModelConfig) -> None:
    cfg = MyAgentConfig(
        chat_model=chat_model,
        system_prompt="p",
        system_prompt_id="id-1",
        max_tokens=1024,
        extra={"max_retries": 3},
    )
    kwargs = cfg.to_kwargs()
    assert kwargs["max_tokens"] == 1024
    assert kwargs["max_retries"] == 3


# --- cache_key ---


def test_cache_key_default_length(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-1")
    assert len(cfg.cache_key()) == 64


def test_cache_key_custom_length(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-1")
    assert len(cfg.cache_key(length=10)) == 10


def test_cache_key_is_deterministic(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"max_retries": 3}
    )
    assert cfg.cache_key() == cfg.cache_key()


def test_cache_key_same_content_same_key(chat_model: ChatModelConfig) -> None:
    cfg1 = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"a": 1, "b": 2}
    )
    cfg2 = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"b": 2, "a": 1}
    )
    assert cfg1.cache_key() == cfg2.cache_key()


def test_cache_key_different_system_prompt_different_key(chat_model: ChatModelConfig) -> None:
    cfg1 = AgentConfig(chat_model=chat_model, system_prompt="p1", system_prompt_id="id-1")
    cfg2 = AgentConfig(chat_model=chat_model, system_prompt="p2", system_prompt_id="id-1")
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_different_system_prompt_id_different_key(chat_model: ChatModelConfig) -> None:
    cfg1 = AgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-1")
    cfg2 = AgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-2")
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_different_chat_model_different_key() -> None:
    cfg1 = AgentConfig(
        chat_model=ChatModelConfig(model="gpt-4"), system_prompt="p", system_prompt_id="id-1"
    )
    cfg2 = AgentConfig(
        chat_model=ChatModelConfig(model="gpt-3.5"), system_prompt="p", system_prompt_id="id-1"
    )
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_same_chat_model_content_same_key() -> None:
    """Two distinct but equal ChatModelConfig objects should produce the
    same AgentConfig cache_key, since cache_key delegates to the nested
    chat_model's own cache_key() rather than object identity."""
    cfg1 = AgentConfig(
        chat_model=ChatModelConfig(model="gpt-4"), system_prompt="p", system_prompt_id="id-1"
    )
    cfg2 = AgentConfig(
        chat_model=ChatModelConfig(model="gpt-4"), system_prompt="p", system_prompt_id="id-1"
    )
    assert cfg1.cache_key() == cfg2.cache_key()


def test_cache_key_reflects_subclass_field(chat_model: ChatModelConfig) -> None:
    cfg1 = MyAgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", max_tokens=1024
    )
    cfg2 = MyAgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", max_tokens=2048
    )
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_does_not_mutate_to_kwargs_result(chat_model: ChatModelConfig) -> None:
    """cache_key() internally overwrites the 'chat_model' entry with a
    hash string; this must not leak back into to_kwargs()'s output."""
    cfg = AgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-1")
    cfg.cache_key()
    assert cfg.to_kwargs()["chat_model"] is chat_model


# --- from_kwargs ---


def test_from_kwargs_basic(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig.from_kwargs(chat_model, "You are helpful.")
    assert cfg.chat_model is chat_model
    assert cfg.system_prompt == "You are helpful."
    assert dict(cfg.extra) == {}


def test_from_kwargs_derives_system_prompt_id_when_omitted(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig.from_kwargs(chat_model, "You are helpful.")
    assert cfg.system_prompt_id == hash_string("You are helpful.")


def test_from_kwargs_uses_explicit_system_prompt_id_when_given(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig.from_kwargs(chat_model, "p", system_prompt_id="custom-id")
    assert cfg.system_prompt_id == "custom-id"


def test_from_kwargs_with_extra_kwargs(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig.from_kwargs(chat_model, "p", max_retries=3)
    assert dict(cfg.extra) == {"max_retries": 3}


def test_from_kwargs_equivalent_to_direct_construction(chat_model: ChatModelConfig) -> None:
    cfg1 = AgentConfig.from_kwargs(chat_model, "p", max_retries=3)
    cfg2 = AgentConfig(
        chat_model=chat_model,
        system_prompt="p",
        system_prompt_id=hash_string("p"),
        extra={"max_retries": 3},
    )
    assert cfg1 == cfg2
    assert cfg1.cache_key() == cfg2.cache_key()


def test_from_kwargs_returns_agent_config_instance(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig.from_kwargs(chat_model, "p")
    assert isinstance(cfg, AgentConfig)


def test_from_kwargs_result_is_frozen(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig.from_kwargs(chat_model, "p")
    with pytest.raises(FrozenInstanceError):
        cfg.system_prompt = "x"


def test_from_kwargs_extra_is_mapping_proxy(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig.from_kwargs(chat_model, "p", max_retries=3)
    assert isinstance(cfg.extra, MappingProxyType)


def test_from_kwargs_duplicate_chat_model_key_raises_type_error(
    chat_model: ChatModelConfig,
) -> None:
    with pytest.raises(TypeError, match=r"got multiple values for argument 'chat_model'"):
        AgentConfig.from_kwargs(chat_model, "p", chat_model=chat_model)


def test_from_kwargs_duplicate_system_prompt_key_raises_type_error(
    chat_model: ChatModelConfig,
) -> None:
    with pytest.raises(TypeError, match=r"got multiple values for argument 'system_prompt'"):
        AgentConfig.from_kwargs(chat_model, "p", system_prompt="other")


def test_from_kwargs_duplicate_system_prompt_id_key_raises_type_error(
    chat_model: ChatModelConfig,
) -> None:
    extra = {"system_prompt_id": "b"}
    with pytest.raises(
        TypeError, match=r"got multiple values for keyword argument 'system_prompt_id'"
    ):
        AgentConfig.from_kwargs(chat_model, "p", system_prompt_id="a", **extra)


def test_from_kwargs_on_subclass_rejects_subclass_specific_field(
    chat_model: ChatModelConfig,
) -> None:
    """from_kwargs's signature only names chat_model/system_prompt/
    system_prompt_id, so it has no way to route max_tokens to the real
    field; it can only go through **kwargs -> extra, which the collision
    guard correctly rejects (accepting it would let extra's max_tokens
    silently override to_kwargs()'s output while cfg.max_tokens itself
    stayed at its default).

    Subclass-specific fields must be set via the regular constructor
    instead.
    """
    with pytest.raises(ValueError, match=r"max_tokens"):
        MyAgentConfig.from_kwargs(chat_model, "p", max_tokens=1024)


def test_from_kwargs_on_subclass_works_for_extra_only(chat_model: ChatModelConfig) -> None:
    """from_kwargs still works on a subclass as long as the kwargs are
    genuine extra data, not a subclass field name."""
    cfg = MyAgentConfig.from_kwargs(chat_model, "p", max_retries=3)
    assert isinstance(cfg, MyAgentConfig)
    assert cfg.max_tokens is None  # unset, uses the field's own default
    assert dict(cfg.extra) == {"max_retries": 3}


# --- __hash__ ---


def test_hash_returns_int(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-1")
    assert isinstance(hash(cfg), int)


def test_hash_equal_for_equal_configs() -> None:
    cfg1 = AgentConfig(
        chat_model=ChatModelConfig(model="gpt-4"), system_prompt="p", system_prompt_id="id-1"
    )
    cfg2 = AgentConfig(
        chat_model=ChatModelConfig(model="gpt-4"), system_prompt="p", system_prompt_id="id-1"
    )
    assert cfg1 == cfg2
    assert hash(cfg1) == hash(cfg2)


def test_hash_differs_for_different_configs(chat_model: ChatModelConfig) -> None:
    cfg1 = AgentConfig(chat_model=chat_model, system_prompt="p1", system_prompt_id="id-1")
    cfg2 = AgentConfig(chat_model=chat_model, system_prompt="p2", system_prompt_id="id-2")
    assert hash(cfg1) != hash(cfg2)


def test_hash_matches_cache_key_hash(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-1")
    assert hash(cfg) == hash(cfg.cache_key())


def test_hash_on_subclass_does_not_raise(chat_model: ChatModelConfig) -> None:
    """Regression test: @dataclass(frozen=True) regenerates __hash__ for
    every dataclass-decorated class unless __hash__ is present in that
    class's own body. MyAgentConfig restates it (delegating to
    AgentConfig.__hash__) specifically so this doesn't fall back to the
    auto-generated version, which would try to hash the unhashable
    `extra` MappingProxyType and raise TypeError."""
    cfg = MyAgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", max_tokens=1024
    )
    assert isinstance(hash(cfg), int)


def test_config_usable_as_dict_key() -> None:
    cfg1 = AgentConfig(
        chat_model=ChatModelConfig(model="gpt-4"), system_prompt="p", system_prompt_id="id-1"
    )
    cfg2 = AgentConfig(
        chat_model=ChatModelConfig(model="gpt-4"), system_prompt="p", system_prompt_id="id-1"
    )
    cache = {cfg1: "cached response"}
    assert cache[cfg2] == "cached response"


def test_config_usable_in_a_set_deduplicates_equal_configs() -> None:
    cfg1 = AgentConfig(
        chat_model=ChatModelConfig(model="gpt-4"), system_prompt="p", system_prompt_id="id-1"
    )
    cfg2 = AgentConfig(
        chat_model=ChatModelConfig(model="gpt-4"), system_prompt="p", system_prompt_id="id-1"
    )
    cfg3 = AgentConfig(
        chat_model=ChatModelConfig(model="gpt-3.5"), system_prompt="p", system_prompt_id="id-1"
    )
    configs = {cfg1, cfg2, cfg3}
    assert len(configs) == 2


# --- subclassing behavior ---


def test_subclass_is_instance_of_agent_config(chat_model: ChatModelConfig) -> None:
    cfg = MyAgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", max_tokens=1024
    )
    assert isinstance(cfg, AgentConfig)


def test_subclass_default_field_value(chat_model: ChatModelConfig) -> None:
    cfg = MyAgentConfig(chat_model=chat_model, system_prompt="p", system_prompt_id="id-1")
    assert cfg.max_tokens is None


def test_subclass_is_frozen(chat_model: ChatModelConfig) -> None:
    cfg = MyAgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", max_tokens=1024
    )
    with pytest.raises(FrozenInstanceError):
        cfg.max_tokens = 2048


# --- repr/str ---


def test_agent_config_repr(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"max_retries": 3}
    )
    assert repr(cfg).startswith("AgentConfig(")


def test_agent_config_str(chat_model: ChatModelConfig) -> None:
    cfg = AgentConfig(
        chat_model=chat_model, system_prompt="p", system_prompt_id="id-1", extra={"max_retries": 3}
    )
    assert str(cfg).startswith("AgentConfig(")
