"""Unit tests for hash_chat_model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr

from zenpyre.hashing import hash_chat_model

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun

MODULE = "zenpyre.hashing.chat_model"


# ---------------------------------------------------------------------------
# Minimal chat model fixtures
# ---------------------------------------------------------------------------


class MinimalChatModel(BaseChatModel):
    model: str = "test-model"
    temperature: float = 0.0

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    @property
    def _llm_type(self) -> str:
        return "minimal"


class AnotherChatModel(BaseChatModel):
    """Same fields as MinimalChatModel but a different class."""

    model: str = "test-model"
    temperature: float = 0.0

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    @property
    def _llm_type(self) -> str:
        return "another"


class ChatModelWithSecret(BaseChatModel):
    model: str = "test-model"
    api_key: SecretStr | None = None

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    @property
    def _llm_type(self) -> str:
        return "with-secret"


class ChatModelWithExcludedFields(BaseChatModel):
    model: str = "test-model"
    default_headers: dict = Field(default_factory=dict)
    default_query: dict = Field(default_factory=dict)

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    @property
    def _llm_type(self) -> str:
        return "with-excluded"


###########################################
#     Tests for hash_chat_model           #
###########################################


# --- Return type and length ---


def test_hash_chat_model_returns_str() -> None:
    assert isinstance(hash_chat_model(MinimalChatModel()), str)


def test_hash_chat_model_default_length() -> None:
    assert len(hash_chat_model(MinimalChatModel())) == 64


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(2, id="min-valid"),
        pytest.param(32, id="middle"),
        pytest.param(64, id="default"),
        pytest.param(128, id="max-valid"),
    ],
)
def test_hash_chat_model_length(length: int) -> None:
    assert len(hash_chat_model(MinimalChatModel(), length=length)) == length


# --- Stability ---


def test_hash_chat_model_same_config_same_hash() -> None:
    assert hash_chat_model(MinimalChatModel()) == hash_chat_model(MinimalChatModel())


def test_hash_chat_model_same_config_same_hash_with_custom_length() -> None:
    assert hash_chat_model(MinimalChatModel(), length=32) == hash_chat_model(
        MinimalChatModel(), length=32
    )


# --- Sensitivity ---


def test_hash_chat_model_different_field_value_different_hash() -> None:
    assert hash_chat_model(MinimalChatModel(temperature=0.0)) != hash_chat_model(
        MinimalChatModel(temperature=1.0)
    )


def test_hash_chat_model_different_class_different_hash() -> None:
    assert hash_chat_model(MinimalChatModel()) != hash_chat_model(AnotherChatModel())


# --- SecretStr handling ---


def test_hash_chat_model_secret_present_vs_absent_different_hash() -> None:
    assert hash_chat_model(ChatModelWithSecret(api_key=SecretStr("sk-test"))) != hash_chat_model(
        ChatModelWithSecret()
    )


def test_hash_chat_model_different_secret_values_same_hash() -> None:
    """Two models with different secret values should hash the same —
    secrets are masked."""
    assert hash_chat_model(ChatModelWithSecret(api_key=SecretStr("sk-aaa"))) == hash_chat_model(
        ChatModelWithSecret(api_key=SecretStr("sk-bbb"))
    )


# --- Excluded fields ---


def test_hash_chat_model_excluded_fields_do_not_affect_hash() -> None:
    llm_a = ChatModelWithExcludedFields(default_headers={"X-Foo": "bar"})
    llm_b = ChatModelWithExcludedFields(default_headers={"X-Foo": "baz"})
    assert hash_chat_model(llm_a) == hash_chat_model(llm_b)


def test_hash_chat_model_default_query_does_not_affect_hash() -> None:
    llm_a = ChatModelWithExcludedFields(default_query={"key": "a"})
    llm_b = ChatModelWithExcludedFields(default_query={"key": "b"})
    assert hash_chat_model(llm_a) == hash_chat_model(llm_b)


# --- length validation ---


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(0, id="zero"),
        pytest.param(1, id="odd-below-min"),
        pytest.param(3, id="odd"),
        pytest.param(-2, id="negative"),
        pytest.param(130, id="above-max"),
    ],
)
def test_hash_chat_model_invalid_length_raises(length: int) -> None:
    with pytest.raises(ValueError, match=r"length"):
        hash_chat_model(MinimalChatModel(), length=length)
