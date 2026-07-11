from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from zenpyre.chat_models import BaseChatModelConfig, ChatModelConfig

#########################################
#     Tests for BaseChatModelConfig     #
#########################################


def test_base_chat_model_config_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
        BaseChatModelConfig()


def test_base_chat_model_config_requires_to_kwargs() -> None:
    class Incomplete(BaseChatModelConfig):
        pass

    with pytest.raises(TypeError, match=r"to_kwargs"):
        Incomplete()


def test_base_chat_model_config_subclass_only_needs_to_kwargs() -> None:
    class Minimal(BaseChatModelConfig):
        def __init__(self, model: str) -> None:
            self.model = model

        def to_kwargs(self) -> dict:
            return {"model": self.model}

    cfg = Minimal("gpt-4")
    assert cfg.to_kwargs() == {"model": "gpt-4"}


def test_base_chat_model_config_cache_key_derived_from_to_kwargs() -> None:
    class Minimal(BaseChatModelConfig):
        def __init__(self, model: str) -> None:
            self.model = model

        def to_kwargs(self) -> dict:
            return {"model": self.model}

    cfg1 = Minimal("gpt-4")
    cfg2 = Minimal("gpt-4")
    cfg3 = Minimal("gpt-3.5")
    assert cfg1.cache_key() == cfg2.cache_key()
    assert cfg1.cache_key() != cfg3.cache_key()


def test_base_chat_model_config_cache_key_respects_length() -> None:
    class Minimal(BaseChatModelConfig):
        def __init__(self, model: str) -> None:
            self.model = model

        def to_kwargs(self) -> dict:
            return {"model": self.model}

    cfg = Minimal("gpt-4")
    assert len(cfg.cache_key()) == 64
    assert len(cfg.cache_key(length=8)) == 8
    assert len(cfg.cache_key(length=16)) == 16


def test_chat_model_config_is_a_base_chat_model_config() -> None:
    cfg = ChatModelConfig(model="gpt-4")
    assert isinstance(cfg, BaseChatModelConfig)


#####################################
#     Tests for ChatModelConfig     #
#####################################

# -- __init__


def test_chat_model_config_default_extra_is_empty() -> None:
    cfg = ChatModelConfig(model="gpt-4")
    assert cfg.model == "gpt-4"
    assert dict(cfg.extra) == {}


def test_chat_model_config_with_extra() -> None:
    cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    assert cfg.model == "gpt-4"
    assert dict(cfg.extra) == {"temperature": 0.2}


def test_chat_model_config_extra_stored_as_mapping_proxy() -> None:
    cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    assert isinstance(cfg.extra, MappingProxyType)


def test_chat_model_config_extra_with_model_key_raises() -> None:
    with pytest.raises(
        ValueError, match=r"'extra' must not contain any of this config's own field names"
    ):
        ChatModelConfig(model="gpt-4", extra={"model": "gpt-3.5"})


def test_chat_model_config_extra_mutation_after_construction_raises() -> None:
    cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    with pytest.raises(TypeError):
        cfg.extra["temperature"] = 0.9


def test_chat_model_config_extra_dict_not_shared_with_caller() -> None:
    """Mutating the original dict passed in should not affect the
    stored, wrapped copy."""
    original = {"temperature": 0.2}
    cfg = ChatModelConfig(model="gpt-4", extra=original)
    original["temperature"] = 0.9
    assert cfg.extra["temperature"] == 0.2


def test_chat_model_config_is_frozen() -> None:
    cfg = ChatModelConfig(model="gpt-4")
    with pytest.raises(FrozenInstanceError):
        cfg.model = "gpt-5"


def test_chat_model_config_frozen_applies_to_extra_reassignment_too() -> None:
    cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    with pytest.raises(FrozenInstanceError):
        cfg.extra = {}


# -- to_kwargs


def test_to_kwargs_model_only() -> None:
    cfg = ChatModelConfig(model="gpt-4")
    assert cfg.to_kwargs() == {"model": "gpt-4"}


def test_to_kwargs_merges_extra() -> None:
    cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2, "max_tokens": 100})
    assert cfg.to_kwargs() == {"model": "gpt-4", "temperature": 0.2, "max_tokens": 100}


def test_to_kwargs_returns_plain_dict() -> None:
    cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    result = cfg.to_kwargs()
    assert type(result) is dict
    assert not isinstance(result, MappingProxyType)


def test_to_kwargs_mutating_result_does_not_affect_config() -> None:
    cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    result = cfg.to_kwargs()
    result["temperature"] = 999
    assert cfg.extra["temperature"] == 0.2


# -- cache_key


def test_cache_key_default_length() -> None:
    cfg = ChatModelConfig(model="gpt-4")
    assert len(cfg.cache_key()) == 64


def test_cache_key_custom_length() -> None:
    cfg = ChatModelConfig(model="gpt-4")
    assert len(cfg.cache_key(length=10)) == 10


def test_cache_key_is_deterministic() -> None:
    cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    assert cfg.cache_key() == cfg.cache_key()


def test_cache_key_same_content_same_key_regardless_of_extra_order() -> None:
    cfg1 = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2, "max_tokens": 100})
    cfg2 = ChatModelConfig(model="gpt-4", extra={"max_tokens": 100, "temperature": 0.2})
    assert cfg1.cache_key() == cfg2.cache_key()


def test_cache_key_different_model_different_key() -> None:
    cfg1 = ChatModelConfig(model="gpt-4")
    cfg2 = ChatModelConfig(model="gpt-3.5")
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_different_extra_different_key() -> None:
    cfg1 = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    cfg2 = ChatModelConfig(model="gpt-4", extra={"temperature": 0.9})
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_no_extra_vs_empty_extra_same_key() -> None:
    cfg1 = ChatModelConfig(model="gpt-4")
    cfg2 = ChatModelConfig(model="gpt-4", extra={})
    assert cfg1.cache_key() == cfg2.cache_key()


# -- from_kwargs


def test_from_kwargs_model_only() -> None:
    cfg = ChatModelConfig.from_kwargs("gpt-4")
    assert cfg.model == "gpt-4"
    assert dict(cfg.extra) == {}


def test_from_kwargs_with_extra_kwargs() -> None:
    cfg = ChatModelConfig.from_kwargs("gpt-4", temperature=0.2, max_tokens=100)
    assert cfg.model == "gpt-4"
    assert dict(cfg.extra) == {"temperature": 0.2, "max_tokens": 100}


def test_from_kwargs_equivalent_to_direct_construction() -> None:
    cfg1 = ChatModelConfig.from_kwargs("gpt-4", temperature=0.2)
    cfg2 = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    assert cfg1 == cfg2
    assert cfg1.cache_key() == cfg2.cache_key()


def test_from_kwargs_returns_chat_model_config_instance() -> None:
    cfg = ChatModelConfig.from_kwargs("gpt-4")
    assert isinstance(cfg, ChatModelConfig)
    assert isinstance(cfg, BaseChatModelConfig)


def test_from_kwargs_duplicate_model_key_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"got multiple values for argument 'model'"):
        ChatModelConfig.from_kwargs("gpt-4", model="gpt-3.5")


def test_from_kwargs_result_is_frozen() -> None:
    cfg = ChatModelConfig.from_kwargs("gpt-4", temperature=0.2)
    with pytest.raises(FrozenInstanceError):
        cfg.model = "gpt-5"


def test_from_kwargs_extra_is_mapping_proxy() -> None:
    cfg = ChatModelConfig.from_kwargs("gpt-4", temperature=0.2)
    assert isinstance(cfg.extra, MappingProxyType)


# -- Tests for ChatModelConfig.__hash__


def test_hash_returns_int() -> None:
    cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    assert isinstance(hash(cfg), int)


def test_hash_equal_for_equal_configs() -> None:
    cfg1 = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    cfg2 = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    assert cfg1 == cfg2
    assert hash(cfg1) == hash(cfg2)


def test_hash_differs_for_different_configs() -> None:
    cfg1 = ChatModelConfig(model="gpt-4")
    cfg2 = ChatModelConfig(model="gpt-3.5")
    assert hash(cfg1) != hash(cfg2)


def test_hash_matches_cache_key_hash() -> None:
    cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    assert hash(cfg) == hash(cfg.cache_key())


def test_config_usable_as_dict_key() -> None:
    cfg1 = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    cfg2 = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    cache = {cfg1: "cached response"}
    assert cache[cfg2] == "cached response"


def test_config_usable_in_a_set_deduplicates_equal_configs() -> None:
    cfg1 = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    cfg2 = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
    cfg3 = ChatModelConfig(model="gpt-3.5", extra={"temperature": 0.2})
    configs = {cfg1, cfg2, cfg3}
    assert len(configs) == 2
