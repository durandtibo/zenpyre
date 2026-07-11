from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from coola.hashing import hash_object

from zenpyre.utils.config import BaseConfig

# ---------------------------------------------------------------------------
# Fixtures / test doubles
# ---------------------------------------------------------------------------


class SimpleConfig(BaseConfig):
    """A minimal concrete subclass implementing only to_kwargs,
    mirroring the pattern shown in BaseConfig's own class docstring."""

    def __init__(self, model: str, extra: dict[str, Any] | None = None) -> None:
        self.model = model
        self.extra = extra or {}

    def to_kwargs(self) -> dict[str, Any]:
        return {"model": self.model, **self.extra}


class WrapperConfig(BaseConfig):
    """A config that nests another BaseConfig as a field, used to test
    cache_key's recursive substitution of nested configs."""

    def __init__(self, inner: Any, label: str = "wrapper") -> None:
        self.inner = inner
        self.label = label

    def to_kwargs(self) -> dict[str, Any]:
        return {"inner": self.inner, "label": self.label}


class ListWrapperConfig(BaseConfig):
    """A config that nests BaseConfig instances inside a list, used to
    test that recursive_apply descends into list-valued fields too."""

    def __init__(self, items: list[Any]) -> None:
        self.items = items

    def to_kwargs(self) -> dict[str, Any]:
        return {"items": self.items}


@pytest.fixture
def config() -> SimpleConfig:
    return SimpleConfig(model="gpt-4")


##################################
#     Tests for BaseConfig      #
##################################


def test_base_config_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseConfig()  # type: ignore[abstract]


def test_subclass_is_instance_of_base_config(config: SimpleConfig) -> None:
    assert isinstance(config, BaseConfig)


def test_subclass_without_to_kwargs_cannot_be_instantiated() -> None:
    class Incomplete(BaseConfig):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# --- to_kwargs ---


def test_to_kwargs_returns_expected_dict(config: SimpleConfig) -> None:
    assert config.to_kwargs() == {"model": "gpt-4"}


def test_to_kwargs_includes_extra_fields() -> None:
    config = SimpleConfig(model="gpt-4", extra={"temperature": 0.2})
    assert config.to_kwargs() == {"model": "gpt-4", "temperature": 0.2}


def test_to_kwargs_returns_plain_dict(config: SimpleConfig) -> None:
    assert type(config.to_kwargs()) is dict


# --- cache_key: basic behavior ---


def test_cache_key_default_length(config: SimpleConfig) -> None:
    assert len(config.cache_key()) == 64


def test_cache_key_custom_length(config: SimpleConfig) -> None:
    assert len(config.cache_key(length=10)) == 10


def test_cache_key_is_deterministic(config: SimpleConfig) -> None:
    assert config.cache_key() == config.cache_key()


def test_cache_key_same_content_same_key() -> None:
    """Field order in the extra dict must not affect the resulting key,
    since cache_key canonicalizes to_kwargs()'s output before
    hashing."""
    cfg1 = SimpleConfig(model="gpt-4", extra={"a": 1, "b": 2})
    cfg2 = SimpleConfig(model="gpt-4", extra={"b": 2, "a": 1})
    assert cfg1.cache_key() == cfg2.cache_key()


def test_cache_key_different_model_different_key() -> None:
    cfg1 = SimpleConfig(model="gpt-4")
    cfg2 = SimpleConfig(model="gpt-3.5")
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_different_extra_different_key() -> None:
    cfg1 = SimpleConfig(model="gpt-4", extra={"temperature": 0.2})
    cfg2 = SimpleConfig(model="gpt-4", extra={"temperature": 0.9})
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_reflects_all_to_kwargs_fields() -> None:
    """A field is only relevant to the cache key insofar as it is
    surfaced by to_kwargs(); this checks the wiring end to end rather
    than asserting anything about a specific unlisted field."""
    cfg1 = SimpleConfig(model="gpt-4", extra={"seed": 1})
    cfg2 = SimpleConfig(model="gpt-4", extra={"seed": 2})
    assert cfg1.to_kwargs() != cfg2.to_kwargs()
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_does_not_mutate_to_kwargs_result(config: SimpleConfig) -> None:
    before = config.to_kwargs()
    config.cache_key()
    assert config.to_kwargs() == before


def test_cache_key_delegates_to_hash_object(config: SimpleConfig) -> None:
    """cache_key must call hash_object with its own to_kwargs() output
    and the requested length, rather than reimplementing hashing
    itself."""
    with patch("zenpyre.utils.config.hash_object", return_value="deadbeef") as mock:
        result = config.cache_key(length=8)
    mock.assert_called_once_with(config.to_kwargs(), length=8)
    assert result == "deadbeef"


# --- cache_key: nested BaseConfig resolution ---


def test_cache_key_nested_config_same_content_same_key() -> None:
    """Two distinct SimpleConfig instances with identical content must
    yield the same key when nested inside a WrapperConfig, since the
    nested config contributes its own cache_key(), not its identity."""
    cfg1 = WrapperConfig(inner=SimpleConfig(model="gpt-4"))
    cfg2 = WrapperConfig(inner=SimpleConfig(model="gpt-4"))
    assert cfg1.cache_key() == cfg2.cache_key()


def test_cache_key_nested_config_different_content_different_key() -> None:
    cfg1 = WrapperConfig(inner=SimpleConfig(model="gpt-4"))
    cfg2 = WrapperConfig(inner=SimpleConfig(model="gpt-3.5"))
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_nested_config_uses_content_not_repr() -> None:
    """Guards against a regression to identity- or repr()-based hashing:

    if the nested config were hashed directly instead of via its own
    cache_key(), two configs with equal content but different object
    identities could still coincidentally differ (or collide) for the
    wrong reason. Asserting the nested key is embedded verifies the
    intended code path is actually taken.
    """
    inner = SimpleConfig(model="gpt-4")
    outer = WrapperConfig(inner=inner)
    assert outer.cache_key() == hash_object(
        {"inner": inner.cache_key(), "label": "wrapper"}, length=64
    )


def test_cache_key_non_nested_field_still_affects_key() -> None:
    """A plain (non-BaseConfig) field alongside a nested config must
    still participate in the hash."""
    inner = SimpleConfig(model="gpt-4")
    cfg1 = WrapperConfig(inner=inner, label="a")
    cfg2 = WrapperConfig(inner=inner, label="b")
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_nested_config_inside_list() -> None:
    """recursive_apply must descend into list-valued fields, not just
    substitute BaseConfig instances that appear as direct dict
    values."""
    cfg1 = ListWrapperConfig(items=[SimpleConfig(model="gpt-4"), "other"])
    cfg2 = ListWrapperConfig(items=[SimpleConfig(model="gpt-4"), "other"])
    cfg3 = ListWrapperConfig(items=[SimpleConfig(model="gpt-3.5"), "other"])
    assert cfg1.cache_key() == cfg2.cache_key()
    assert cfg1.cache_key() != cfg3.cache_key()


def test_cache_key_deeply_nested_configs() -> None:
    """Nesting a WrapperConfig inside another WrapperConfig must resolve
    at every level, not just one level deep."""
    cfg1 = WrapperConfig(inner=WrapperConfig(inner=SimpleConfig(model="gpt-4")))
    cfg2 = WrapperConfig(inner=WrapperConfig(inner=SimpleConfig(model="gpt-4")))
    cfg3 = WrapperConfig(inner=WrapperConfig(inner=SimpleConfig(model="gpt-3.5")))
    assert cfg1.cache_key() == cfg2.cache_key()
    assert cfg1.cache_key() != cfg3.cache_key()


def test_cache_key_nested_config_does_not_mutate_to_kwargs_result() -> None:
    """The substitution must not leak back into to_kwargs(): the nested
    field should still be the live BaseConfig instance afterward, not a
    string left over from a previous cache_key() call."""
    inner = SimpleConfig(model="gpt-4")
    outer = WrapperConfig(inner=inner)
    outer.cache_key()
    assert outer.to_kwargs()["inner"] is inner
