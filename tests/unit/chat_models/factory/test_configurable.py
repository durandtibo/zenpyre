from __future__ import annotations

from coola.equality import objects_are_equal
from langchain_core.language_models import BaseChatModel, FakeListChatModel

from zenpyre.chat_models.factory import (
    BaseChatModelFactory,
    ConfigurableChatModelFactory,
)

FAKE_LIST_CHAT_MODEL_TARGET = "langchain_core.language_models.FakeListChatModel"


def _make_chat_model() -> FakeListChatModel:
    """Return a FakeListChatModel instance for testing."""
    return FakeListChatModel(responses=["hello"])


##################################################
#     Tests for ConfigurableChatModelFactory     #
##################################################


# --- Inheritance ---


def test_configurable_chat_model_factory_is_base_chat_model_factory() -> None:
    assert isinstance(ConfigurableChatModelFactory(_make_chat_model()), BaseChatModelFactory)


# --- make_chat_model from instance ---


def test_configurable_chat_model_factory_make_chat_model_returns_base_chat_model() -> None:
    factory = ConfigurableChatModelFactory(_make_chat_model())
    assert isinstance(factory.make_chat_model(), BaseChatModel)


def test_configurable_chat_model_factory_make_chat_model_returns_same_instance() -> None:
    chat_model = _make_chat_model()
    factory = ConfigurableChatModelFactory(chat_model)
    assert factory.make_chat_model() is chat_model


# --- make_chat_model from dict ---


def test_configurable_chat_model_factory_make_chat_model_from_dict_returns_base_chat_model() -> (
    None
):
    factory = ConfigurableChatModelFactory(
        {"_target_": FAKE_LIST_CHAT_MODEL_TARGET, "responses": ["hello"]}
    )
    assert isinstance(factory.make_chat_model(), BaseChatModel)


def test_configurable_chat_model_factory_make_chat_model_from_dict_returns_correct_type() -> None:
    factory = ConfigurableChatModelFactory(
        {"_target_": FAKE_LIST_CHAT_MODEL_TARGET, "responses": ["hello"]}
    )
    assert isinstance(factory.make_chat_model(), FakeListChatModel)


# --- _get_repr_kwargs ---


def test_configurable_chat_model_factory_get_repr_kwargs_instance() -> None:
    chat_model = _make_chat_model()
    factory = ConfigurableChatModelFactory(chat_model)
    assert objects_are_equal(factory._get_repr_kwargs(), {"chat_model": chat_model})


def test_configurable_chat_model_factory_get_repr_kwargs_dict_input() -> None:
    config = {"_target_": FAKE_LIST_CHAT_MODEL_TARGET, "responses": ["hello"]}
    factory = ConfigurableChatModelFactory(config)
    assert objects_are_equal(factory._get_repr_kwargs(), {"chat_model": config})


# --- __repr__ and __str__ ---


def test_configurable_chat_model_factory_repr_starts_with_class_name() -> None:
    factory = ConfigurableChatModelFactory(_make_chat_model())
    assert repr(factory).startswith("ConfigurableChatModelFactory(")


def test_configurable_chat_model_factory_str_starts_with_class_name() -> None:
    factory = ConfigurableChatModelFactory(_make_chat_model())
    assert str(factory).startswith("ConfigurableChatModelFactory(")


def test_configurable_chat_model_factory_repr_contains_chat_model() -> None:
    factory = ConfigurableChatModelFactory(_make_chat_model())
    assert "chat_model" in repr(factory)


def test_configurable_chat_model_factory_str_contains_chat_model() -> None:
    factory = ConfigurableChatModelFactory(_make_chat_model())
    assert "chat_model" in str(factory)
