from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from types import MappingProxyType

import pytest

from zenpyre.utils.config import BaseConfig, ExtraFieldsConfig

# ---------------------------------------------------------------------------
# Fixtures / test doubles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DummyConfig(ExtraFieldsConfig):
    """A minimal concrete subclass adding one typed field, mirroring the
    pattern subclasses are expected to follow (declare fields, restate
    __hash__, nothing else)."""

    name: str = "dummy"

    __hash__ = ExtraFieldsConfig.__hash__


@dataclass(frozen=True)
class OtherFieldConfig(ExtraFieldsConfig):
    """A second, distinct concrete subclass with a different field name,
    used to confirm the collision guard is driven by each subclass's own
    fields rather than a hardcoded name."""

    max_tokens: int | None = None

    __hash__ = ExtraFieldsConfig.__hash__


@pytest.fixture
def config() -> DummyConfig:
    return DummyConfig(name="a")


#######################################
#     Tests for ExtraFieldsConfig     #
#######################################


def test_extra_fields_config_is_directly_instantiable() -> None:
    """Unlike BaseConfig, ExtraFieldsConfig is not abstract: it already
    implements to_kwargs() itself (returning just the dataclass fields
    plus extra, i.e. only `extra` when used directly), which is
    precisely what lets subclasses skip implementing it."""
    config = ExtraFieldsConfig(extra={"x": 1})
    assert config.to_kwargs() == {"x": 1}


def test_subclass_is_instance_of_extra_fields_config(config: DummyConfig) -> None:
    assert isinstance(config, ExtraFieldsConfig)


def test_subclass_is_instance_of_base_config(config: DummyConfig) -> None:
    assert isinstance(config, BaseConfig)


# --- construction & extra ---


def test_default_extra_is_empty(config: DummyConfig) -> None:
    assert dict(config.extra) == {}


def test_extra_stores_given_values() -> None:
    config = DummyConfig(name="a", extra={"max_retries": 3})
    assert dict(config.extra) == {"max_retries": 3}


def test_extra_stored_as_mapping_proxy(config: DummyConfig) -> None:
    assert isinstance(config.extra, MappingProxyType)


def test_extra_dict_not_shared_with_caller() -> None:
    original = {"max_retries": 3}
    config = DummyConfig(name="a", extra=original)
    original["max_retries"] = 999
    assert config.extra["max_retries"] == 3


def test_extra_mutation_after_construction_raises(config: DummyConfig) -> None:
    config = DummyConfig(name="a", extra={"max_retries": 3})
    with pytest.raises(TypeError):
        config.extra["max_retries"] = 999


# --- frozen ---


def test_config_is_frozen(config: DummyConfig) -> None:
    with pytest.raises(FrozenInstanceError):
        config.name = "other"


def test_frozen_applies_to_extra_too(config: DummyConfig) -> None:
    with pytest.raises(FrozenInstanceError):
        config.extra = {}


# --- field/extra collision guard ---


def test_extra_with_own_field_name_raises() -> None:
    with pytest.raises(
        ValueError, match=r"'extra' must not contain any of this config's own field names"
    ):
        DummyConfig(name="a", extra={"name": "x"})


def test_extra_with_unrelated_key_does_not_raise() -> None:
    DummyConfig(name="a", extra={"max_retries": 3})  # should not raise


def test_collision_error_lists_reserved_names() -> None:
    with pytest.raises(ValueError, match=r"name") as exc_info:
        DummyConfig(name="a", extra={"name": "x"})
    assert "name" in str(exc_info.value)


def test_collision_guard_is_driven_by_subclass_fields_not_hardcoded() -> None:
    """A different subclass with a different field name must guard its
    own field, confirming the check isn't hardcoded to any one name
    (e.g. leftover from copy-pasting a specific subclass's field)."""
    with pytest.raises(ValueError, match=r"max_tokens"):
        OtherFieldConfig(max_tokens=1, extra={"max_tokens": 999})


def test_collision_guard_does_not_flag_extra_itself() -> None:
    """'extra' is the mechanism, not a reserved data field, so it must
    not appear in the reserved-name set used for the collision check."""
    DummyConfig(name="a", extra={"extra": "should not collide"})  # should not raise


# --- to_kwargs ---


def test_to_kwargs_contains_typed_field(config: DummyConfig) -> None:
    assert config.to_kwargs()["name"] == "a"


def test_to_kwargs_merges_extra() -> None:
    config = DummyConfig(name="a", extra={"max_retries": 3})
    assert config.to_kwargs()["max_retries"] == 3


def test_to_kwargs_does_not_include_extra_key_itself() -> None:
    config = DummyConfig(name="a", extra={"max_retries": 3})
    assert "extra" not in config.to_kwargs()


def test_to_kwargs_returns_plain_dict(config: DummyConfig) -> None:
    assert type(config.to_kwargs()) is dict


def test_to_kwargs_mutating_result_does_not_affect_config() -> None:
    config = DummyConfig(name="a", extra={"max_retries": 3})
    kwargs = config.to_kwargs()
    kwargs["max_retries"] = 999
    assert config.extra["max_retries"] == 3


def test_to_kwargs_subclass_field_is_automatically_included() -> None:
    """The key behavioral guarantee of the shared base: a subclass's
    own typed field must appear in to_kwargs() via introspection,
    without the subclass overriding to_kwargs() itself."""
    config = OtherFieldConfig(max_tokens=1024)
    kwargs = config.to_kwargs()
    assert kwargs["max_tokens"] == 1024


def test_to_kwargs_subclass_field_alongside_extra() -> None:
    config = OtherFieldConfig(max_tokens=1024, extra={"max_retries": 3})
    kwargs = config.to_kwargs()
    assert kwargs["max_tokens"] == 1024
    assert kwargs["max_retries"] == 3


# --- cache_key ---


def test_cache_key_default_length(config: DummyConfig) -> None:
    assert len(config.cache_key()) == 64


def test_cache_key_custom_length(config: DummyConfig) -> None:
    assert len(config.cache_key(length=10)) == 10


def test_cache_key_is_deterministic(config: DummyConfig) -> None:
    assert config.cache_key() == config.cache_key()


def test_cache_key_same_content_same_key() -> None:
    cfg1 = DummyConfig(name="a", extra={"x": 1, "y": 2})
    cfg2 = DummyConfig(name="a", extra={"y": 2, "x": 1})
    assert cfg1.cache_key() == cfg2.cache_key()


def test_cache_key_different_field_different_key() -> None:
    cfg1 = DummyConfig(name="a")
    cfg2 = DummyConfig(name="b")
    assert cfg1.cache_key() != cfg2.cache_key()


def test_cache_key_different_extra_different_key() -> None:
    cfg1 = DummyConfig(name="a", extra={"x": 1})
    cfg2 = DummyConfig(name="a", extra={"x": 2})
    assert cfg1.cache_key() != cfg2.cache_key()


# --- __hash__ ---


def test_hash_returns_int(config: DummyConfig) -> None:
    assert isinstance(hash(config), int)


def test_hash_equal_for_equal_configs() -> None:
    cfg1 = DummyConfig(name="a")
    cfg2 = DummyConfig(name="a")
    assert cfg1 == cfg2
    assert hash(cfg1) == hash(cfg2)


def test_hash_differs_for_different_configs() -> None:
    cfg1 = DummyConfig(name="a")
    cfg2 = DummyConfig(name="b")
    assert hash(cfg1) != hash(cfg2)


def test_hash_matches_cache_key_hash(config: DummyConfig) -> None:
    assert hash(config) == hash(config.cache_key())


def test_hash_does_not_raise_on_unhashable_extra() -> None:
    """The whole point of the custom __hash__: the auto-generated
    dataclass hash would try to hash `extra` (a MappingProxyType,
    itself unhashable) directly and fail. This must not happen."""
    config = DummyConfig(name="a", extra={"max_retries": 3})
    hash(config)  # should not raise


def test_config_usable_as_dict_key() -> None:
    cfg1 = DummyConfig(name="a")
    cfg2 = DummyConfig(name="a")
    cache = {cfg1: "cached response"}
    assert cache[cfg2] == "cached response"


def test_config_usable_in_a_set_deduplicates_equal_configs() -> None:
    cfg1 = DummyConfig(name="a")
    cfg2 = DummyConfig(name="a")
    cfg3 = DummyConfig(name="b")
    configs = {cfg1, cfg2, cfg3}
    assert len(configs) == 2


# --- kw_only field ordering ---


def test_subclass_field_without_default_does_not_raise_at_class_definition() -> None:
    """Because ExtraFieldsConfig declares `extra` as keyword-only, a
    subclass may add a non-defaulted field without triggering
    dataclasses' 'non-default argument follows default argument' error,
    even though `extra` (with its default) is defined first, in the
    parent."""

    @dataclass(frozen=True)
    class NoDefaultFieldConfig(ExtraFieldsConfig):
        required_field: str  # no default; must not raise at class body evaluation

        __hash__ = ExtraFieldsConfig.__hash__

    cfg = NoDefaultFieldConfig(required_field="value")
    assert cfg.required_field == "value"


def test_extra_must_be_passed_by_keyword() -> None:
    """Extra is keyword-only, so passing it positionally must fail."""
    with pytest.raises(TypeError):
        DummyConfig("a", {"x": 1})  # type: ignore[misc]


# --- __hash__ shadowing guard (regression coverage) ---


def test_forgetting_to_restate_hash_reintroduces_the_bug() -> None:
    """Documents *why* subclasses must restate __hash__: without it,
    @dataclass(frozen=True) generates a fresh __hash__ for the
    subclass that tries to hash the unhashable `extra` field,
    overriding the inherited working implementation."""

    @dataclass(frozen=True)
    class ForgotHashConfig(ExtraFieldsConfig):
        name: str = "x"
        # deliberately omitted: __hash__ = ExtraFieldsConfig.__hash__

    config = ForgotHashConfig(name="a", extra={"max_retries": 3})
    with pytest.raises(TypeError):
        hash(config)


def test_restating_hash_fixes_the_regression() -> None:
    @dataclass(frozen=True)
    class RestatedHashConfig(ExtraFieldsConfig):
        name: str = "x"
        __hash__ = ExtraFieldsConfig.__hash__

    config = RestatedHashConfig(name="a", extra={"max_retries": 3})
    assert hash(config) == hash(config.cache_key())
