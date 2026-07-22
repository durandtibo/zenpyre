from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from coola.equality import objects_are_equal
from langchain_core.language_models import FakeListChatModel
from persista.cache import Cache

from zenpyre.chat_models.factory import BaseChatModelFactory, CachingChatModelFactory
from zenpyre.utils.config import Config

MODULE = "zenpyre.chat_models.factory.cache"

MINIMAL_CHAT_MODEL_FACTORY_TARGET = (
    "tests.unit.chat_models.factory.test_cache.MinimalChatModelFactory"
)


class MinimalChatModelFactory(BaseChatModelFactory):
    """Minimal concrete BaseChatModelFactory for testing."""

    def make_chat_model(self) -> Any:
        return FakeListChatModel(responses=["hello"])


def _make_chat_model_factory() -> MagicMock:
    """Return a MagicMock standing in for a BaseChatModelFactory."""
    return MagicMock(spec=BaseChatModelFactory)


def _make_factory(**overrides: Any) -> CachingChatModelFactory:
    """Return a CachingChatModelFactory instance for testing."""
    kwargs = {
        "chat_model_factory": _make_chat_model_factory(),
        "cache": None,
        "key_fn": None,
        "ignore_none": False,
    }
    kwargs.update(overrides)
    return CachingChatModelFactory(**kwargs)


#############################################
#     Tests for CachingChatModelFactory     #
#############################################


# --- Inheritance ---


def test_caching_chat_model_factory_is_base_chat_model_factory() -> None:
    assert isinstance(_make_factory(), BaseChatModelFactory)


# --- __init__ stores args ---


def test_caching_chat_model_factory_stores_chat_model_factory() -> None:
    chat_model_factory = _make_chat_model_factory()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    assert factory._chat_model_factory is chat_model_factory


def test_caching_chat_model_factory_stores_cache() -> None:
    cache = Cache()
    factory = _make_factory(cache=cache)
    assert factory._cache is cache


def test_caching_chat_model_factory_default_cache_is_none() -> None:
    factory = _make_factory()
    assert factory._cache is None


def test_caching_chat_model_factory_stores_key_fn() -> None:
    key_fn = lambda x: str(x)  # noqa: E731
    factory = _make_factory(key_fn=key_fn)
    assert factory._key_fn is key_fn


def test_caching_chat_model_factory_default_ignore_none_is_false() -> None:
    factory = _make_factory()
    assert factory._ignore_none is False


def test_caching_chat_model_factory_stores_ignore_none_true() -> None:
    factory = _make_factory(ignore_none=True)
    assert factory._ignore_none is True


# --- __init__ resolves chat_model_factory ---


def test_caching_chat_model_factory_resolves_chat_model_factory_from_dict() -> None:
    factory = _make_factory(chat_model_factory={"_target_": MINIMAL_CHAT_MODEL_FACTORY_TARGET})
    assert isinstance(factory._chat_model_factory, MinimalChatModelFactory)


def test_caching_chat_model_factory_resolves_chat_model_factory_from_base_config() -> None:
    config = Config.from_kwargs(_target_=MINIMAL_CHAT_MODEL_FACTORY_TARGET)
    factory = _make_factory(chat_model_factory=config)
    assert isinstance(factory._chat_model_factory, MinimalChatModelFactory)


def test_caching_chat_model_factory_invalid_chat_model_factory_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BaseChatModelFactory instance"):
        _make_factory(chat_model_factory="not-a-chat-model-factory")


# --- make_chat_model wiring ---


def test_caching_chat_model_factory_make_chat_model_builds_chat_model_from_factory() -> None:
    chat_model_factory = _make_chat_model_factory()
    factory = _make_factory(chat_model_factory=chat_model_factory)
    with patch(f"{MODULE}.CachingChatModel"):
        factory.make_chat_model()
        chat_model_factory.make_chat_model.assert_called_once_with()


def test_caching_chat_model_factory_make_chat_model_wraps_in_caching_chat_model() -> None:
    chat_model_factory = _make_chat_model_factory()
    cache = Cache()
    key_fn = lambda x: str(x)  # noqa: E731
    factory = _make_factory(
        chat_model_factory=chat_model_factory,
        cache=cache,
        key_fn=key_fn,
        ignore_none=True,
    )
    with patch(f"{MODULE}.CachingChatModel") as mock_caching_chat_model_cls:
        factory.make_chat_model()
        mock_caching_chat_model_cls.assert_called_once_with(
            chat_model=chat_model_factory.make_chat_model.return_value,
            result_cache=cache,
            key_fn=key_fn,
            ignore_none=True,
        )


def test_caching_chat_model_factory_make_chat_model_returns_caching_chat_model() -> None:
    factory = _make_factory()
    with patch(f"{MODULE}.CachingChatModel") as mock_caching_chat_model_cls:
        result = factory.make_chat_model()
        assert result is mock_caching_chat_model_cls.return_value


# --- _get_repr_kwargs ---


def test_caching_chat_model_factory_get_repr_kwargs() -> None:
    chat_model_factory = _make_chat_model_factory()
    cache = Cache()
    key_fn = lambda x: str(x)  # noqa: E731
    factory = _make_factory(
        chat_model_factory=chat_model_factory,
        cache=cache,
        key_fn=key_fn,
        ignore_none=True,
    )
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {
            "chat_model_factory": chat_model_factory,
            "cache": cache,
            "key_fn": key_fn,
            "ignore_none": True,
        },
    )


# --- __repr__ and __str__ ---


def test_caching_chat_model_factory_repr_starts_with_class_name() -> None:
    factory = _make_factory()
    assert repr(factory).startswith("CachingChatModelFactory(")


def test_caching_chat_model_factory_str_starts_with_class_name() -> None:
    factory = _make_factory()
    assert str(factory).startswith("CachingChatModelFactory(")
