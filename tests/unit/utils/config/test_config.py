from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from zenpyre.utils.config import Config

#############################
#     Tests for Config      #
#############################


# --- construction / __post_init__ ---


def test_default_extra_is_empty() -> None:
    config = Config()
    assert dict(config.extra) == {}


def test_extra_is_stored() -> None:
    config = Config(extra={"a": 1, "b": 2})
    assert dict(config.extra) == {"a": 1, "b": 2}


def test_extra_is_read_only() -> None:
    config = Config(extra={"a": 1})
    with pytest.raises(TypeError):
        config.extra["a"] = 2


def test_config_is_frozen() -> None:
    config = Config()
    with pytest.raises(FrozenInstanceError):
        config.extra = {"a": 1}


# --- get_value (inherited) ---


def test_get_value_returns_extra_entry() -> None:
    config = Config(extra={"a": 1})
    assert config.get_value("a") == 1


def test_get_value_missing_raises_key_error() -> None:
    config = Config()
    with pytest.raises(KeyError):
        config.get_value("missing")


def test_get_value_missing_returns_default() -> None:
    config = Config()
    assert config.get_value("missing", default=42) == 42


def test_get_value_default_none_is_respected() -> None:
    config = Config()
    assert config.get_value("missing", default=None) is None


# --- to_kwargs (inherited) ---


def test_to_kwargs_empty() -> None:
    config = Config()
    assert config.to_kwargs() == {}


def test_to_kwargs_includes_extra() -> None:
    config = Config(extra={"a": 1, "b": 2})
    assert config.to_kwargs() == {"a": 1, "b": 2}


def test_to_kwargs_returns_new_dict_each_call() -> None:
    config = Config(extra={"a": 1})
    result1 = config.to_kwargs()
    result2 = config.to_kwargs()
    assert result1 == result2
    assert result1 is not result2


# --- from_kwargs (inherited) ---


def test_from_kwargs_routes_all_keys_to_extra() -> None:
    config = Config.from_kwargs(a=1, b=2)
    assert dict(config.extra) == {"a": 1, "b": 2}


def test_from_kwargs_no_kwargs() -> None:
    config = Config.from_kwargs()
    assert dict(config.extra) == {}


def test_from_kwargs_round_trips_with_to_kwargs() -> None:
    config = Config.from_kwargs(a=1, b=2)
    assert config.to_kwargs() == {"a": 1, "b": 2}


# --- equality (inherited/generated) ---


def test_equal_configs_are_equal() -> None:
    config1 = Config(extra={"a": 1})
    config2 = Config(extra={"a": 1})
    assert config1 == config2


def test_different_configs_are_not_equal() -> None:
    config1 = Config(extra={"a": 1})
    config2 = Config(extra={"a": 2})
    assert config1 != config2


def test_config_not_equal_to_other_type() -> None:
    config = Config()
    assert config != object()


# --- __hash__ (own override) ---


def test_config_is_hashable() -> None:
    config = Config(extra={"a": 1})
    hash(config)  # should not raise


def test_equal_configs_have_equal_hash() -> None:
    config1 = Config(extra={"a": 1})
    config2 = Config(extra={"a": 1})
    assert hash(config1) == hash(config2)


def test_config_hash_delegates_to_extra_fields_config_hash() -> None:
    from zenpyre.utils.config import ExtraFieldsConfig

    config = Config(extra={"a": 1})
    assert hash(config) == ExtraFieldsConfig.__hash__(config)


def test_config_can_be_used_in_a_set() -> None:
    config1 = Config(extra={"a": 1})
    config2 = Config(extra={"a": 1})
    config3 = Config(extra={"a": 2})
    assert {config1, config2, config3} == {config1, config3}


# --- cache_key (inherited from BaseConfig) ---


def test_cache_key_is_a_string() -> None:
    config = Config(extra={"a": 1})
    assert isinstance(config.cache_key(), str)


def test_cache_key_is_deterministic_for_equal_configs() -> None:
    config1 = Config(extra={"a": 1})
    config2 = Config(extra={"a": 1})
    assert config1.cache_key() == config2.cache_key()


def test_cache_key_differs_for_different_configs() -> None:
    config1 = Config(extra={"a": 1})
    config2 = Config(extra={"a": 2})
    assert config1.cache_key() != config2.cache_key()
