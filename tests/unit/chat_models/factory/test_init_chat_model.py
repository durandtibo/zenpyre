from __future__ import annotations

from typing import Any
from unittest.mock import patch

from coola.equality import objects_are_equal

from zenpyre.chat_models.factory import BaseChatModelFactory, InitChatModelFactory

MODULE = "zenpyre.chat_models.factory.init_chat_model"


def _make_factory(**overrides: Any) -> InitChatModelFactory:
    """Return an InitChatModelFactory instance for testing."""
    kwargs = {"model": "openai:gpt-4o-mini"}
    kwargs.update(overrides)
    return InitChatModelFactory(**kwargs)


##########################################
#     Tests for InitChatModelFactory     #
##########################################


# --- Inheritance ---


def test_init_chat_model_factory_is_base_chat_model_factory() -> None:
    assert isinstance(_make_factory(), BaseChatModelFactory)


# --- __init__ stores args ---


def test_init_chat_model_factory_stores_model() -> None:
    factory = _make_factory(model="openai:gpt-4o")
    assert factory._model == "openai:gpt-4o"


def test_init_chat_model_factory_default_model_is_none() -> None:
    factory = InitChatModelFactory()
    assert factory._model is None


def test_init_chat_model_factory_default_kwargs_is_empty() -> None:
    factory = _make_factory()
    assert factory._kwargs == {}


def test_init_chat_model_factory_stores_kwargs() -> None:
    factory = _make_factory(temperature=0.2, model_provider="openai")
    assert factory._kwargs == {"temperature": 0.2, "model_provider": "openai"}


# --- make_chat_model wiring ---


def test_init_chat_model_factory_make_chat_model_calls_init_chat_model() -> None:
    factory = _make_factory(model="openai:gpt-4o-mini", temperature=0.2)
    with patch(f"{MODULE}.init_chat_model") as mock_init_chat_model:
        factory.make_chat_model()
        mock_init_chat_model.assert_called_once_with("openai:gpt-4o-mini", temperature=0.2)


def test_init_chat_model_factory_make_chat_model_calls_init_chat_model_with_no_extra_kwargs() -> (
    None
):
    factory = _make_factory(model="openai:gpt-4o-mini")
    with patch(f"{MODULE}.init_chat_model") as mock_init_chat_model:
        factory.make_chat_model()
        mock_init_chat_model.assert_called_once_with("openai:gpt-4o-mini")


def test_init_chat_model_factory_make_chat_model_returns_init_chat_model_result() -> None:
    factory = _make_factory()
    with patch(f"{MODULE}.init_chat_model") as mock_init_chat_model:
        result = factory.make_chat_model()
        assert result is mock_init_chat_model.return_value


def test_init_chat_model_factory_make_chat_model_builds_new_model_each_call() -> None:
    factory = _make_factory()
    with patch(f"{MODULE}.init_chat_model") as mock_init_chat_model:
        factory.make_chat_model()
        factory.make_chat_model()
        assert mock_init_chat_model.call_count == 2


# --- _get_repr_kwargs ---


def test_init_chat_model_factory_get_repr_kwargs() -> None:
    factory = _make_factory(model="openai:gpt-4o-mini", temperature=0.2)
    assert objects_are_equal(
        factory._get_repr_kwargs(), {"model": "openai:gpt-4o-mini", "temperature": 0.2}
    )


def test_init_chat_model_factory_get_repr_kwargs_with_no_extra_kwargs() -> None:
    factory = _make_factory(model="openai:gpt-4o-mini")
    assert objects_are_equal(factory._get_repr_kwargs(), {"model": "openai:gpt-4o-mini"})


# --- __repr__ and __str__ ---


def test_init_chat_model_factory_repr_starts_with_class_name() -> None:
    factory = _make_factory()
    assert repr(factory).startswith("InitChatModelFactory(")


def test_init_chat_model_factory_str_starts_with_class_name() -> None:
    factory = _make_factory()
    assert str(factory).startswith("InitChatModelFactory(")


def test_init_chat_model_factory_repr_contains_model() -> None:
    factory = _make_factory()
    assert "model" in repr(factory)


def test_init_chat_model_factory_str_contains_model() -> None:
    factory = _make_factory()
    assert "model" in str(factory)
