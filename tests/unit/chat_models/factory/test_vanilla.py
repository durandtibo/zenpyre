from __future__ import annotations

from coola.equality import objects_are_equal
from langchain_core.language_models import BaseChatModel, FakeListChatModel

from zenpyre.chat_models.factory import (
    BaseChatModelFactory,
    ChatModelFactory,
)


def _make_chat_model() -> FakeListChatModel:
    """Return a FakeListChatModel instance for testing."""
    return FakeListChatModel(responses=["hello"])


##################################################
#     Tests for ChatModelFactory                #
##################################################


# --- Inheritance ---


def test_chat_model_factory_is_base_chat_model_factory() -> None:
    assert isinstance(ChatModelFactory(_make_chat_model()), BaseChatModelFactory)


# --- make_chat_model ---


def test_chat_model_factory_make_chat_model_returns_base_chat_model() -> None:
    factory = ChatModelFactory(_make_chat_model())
    assert isinstance(factory.make_chat_model(), BaseChatModel)


def test_chat_model_factory_make_chat_model_returns_same_instance() -> None:
    chat_model = _make_chat_model()
    factory = ChatModelFactory(chat_model)
    assert factory.make_chat_model() is chat_model


def test_chat_model_factory_make_chat_model_returns_same_instance_across_calls() -> None:
    chat_model = _make_chat_model()
    factory = ChatModelFactory(chat_model)
    assert factory.make_chat_model() is factory.make_chat_model()


# --- _get_repr_kwargs ---


def test_chat_model_factory_get_repr_kwargs() -> None:
    chat_model = _make_chat_model()
    factory = ChatModelFactory(chat_model)
    assert objects_are_equal(factory._get_repr_kwargs(), {"chat_model": chat_model})


# --- __repr__ and __str__ ---


def test_chat_model_factory_repr_starts_with_class_name() -> None:
    factory = ChatModelFactory(_make_chat_model())
    assert repr(factory).startswith("ChatModelFactory(")


def test_chat_model_factory_str_starts_with_class_name() -> None:
    factory = ChatModelFactory(_make_chat_model())
    assert str(factory).startswith("ChatModelFactory(")


def test_chat_model_factory_repr_contains_chat_model() -> None:
    factory = ChatModelFactory(_make_chat_model())
    assert "chat_model" in repr(factory)


def test_chat_model_factory_str_contains_chat_model() -> None:
    factory = ChatModelFactory(_make_chat_model())
    assert "chat_model" in str(factory)
