from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel, FakeListChatModel

from zenpyre.chat_models import resolve_chat_model
from zenpyre.utils.config import Config

FAKE_LIST_CHAT_MODEL_TARGET = "langchain_core.language_models.FakeListChatModel"


def _make_chat_model() -> FakeListChatModel:
    """Return a FakeListChatModel instance for testing."""
    return FakeListChatModel(responses=["hello"])


########################################
#     Tests for resolve_chat_model     #
########################################


# --- Pass-through ---


def test_resolve_chat_model_returns_base_chat_model_instance() -> None:
    assert isinstance(resolve_chat_model(_make_chat_model()), BaseChatModel)


def test_resolve_chat_model_passthrough_returns_same_instance() -> None:
    chat_model = _make_chat_model()
    assert resolve_chat_model(chat_model) is chat_model


# --- From dict ---


def test_resolve_chat_model_from_dict_returns_base_chat_model() -> None:
    result = resolve_chat_model({"_target_": FAKE_LIST_CHAT_MODEL_TARGET, "responses": ["hello"]})
    assert isinstance(result, BaseChatModel)


def test_resolve_chat_model_from_dict_returns_correct_type() -> None:
    result = resolve_chat_model({"_target_": FAKE_LIST_CHAT_MODEL_TARGET, "responses": ["hello"]})
    assert isinstance(result, FakeListChatModel)


# --- From BaseConfig ---


def test_resolve_chat_model_from_base_config_returns_base_chat_model() -> None:
    config = Config.from_kwargs(_target_=FAKE_LIST_CHAT_MODEL_TARGET, responses=["hello"])
    result = resolve_chat_model(config)
    assert isinstance(result, BaseChatModel)


def test_resolve_chat_model_from_base_config_returns_correct_type() -> None:
    config = Config.from_kwargs(_target_=FAKE_LIST_CHAT_MODEL_TARGET, responses=["hello"])
    result = resolve_chat_model(config)
    assert isinstance(result, FakeListChatModel)


# --- Invalid input ---


def test_resolve_chat_model_invalid_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BaseChatModel instance"):
        resolve_chat_model("not-a-chat-model")
