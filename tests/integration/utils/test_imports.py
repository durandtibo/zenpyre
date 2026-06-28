from __future__ import annotations

import pytest

from zenpyre.testing.fixtures import (
    langchain_anthropic_available,
    langchain_anthropic_not_available,
    langchain_google_genai_available,
    langchain_google_genai_not_available,
    langchain_ollama_available,
    langchain_ollama_not_available,
    langchain_openai_available,
    langchain_openai_not_available,
)
from zenpyre.utils.imports import (
    check_langchain_anthropic,
    check_langchain_google_genai,
    check_langchain_ollama,
    check_langchain_openai,
    is_langchain_anthropic_available,
    is_langchain_google_genai_available,
    is_langchain_ollama_available,
    is_langchain_openai_available,
)

###############################
#     langchain_anthropic     #
###############################


@langchain_anthropic_available
def test_check_langchain_anthropic_with_package() -> None:
    check_langchain_anthropic()


@langchain_anthropic_not_available
def test_check_langchain_anthropic_without_package() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_anthropic' package is required but not installed."
    ):
        check_langchain_anthropic()


@langchain_anthropic_available
def test_is_langchain_anthropic_available_true() -> None:
    assert is_langchain_anthropic_available()


@langchain_anthropic_not_available
def test_is_langchain_anthropic_available_false() -> None:
    assert not is_langchain_anthropic_available()


##################################
#     langchain_google_genai     #
##################################


@langchain_google_genai_available
def test_check_langchain_google_genai_with_package() -> None:
    check_langchain_google_genai()


@langchain_google_genai_not_available
def test_check_langchain_google_genai_without_package() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_google_genai' package is required but not installed."
    ):
        check_langchain_google_genai()


@langchain_google_genai_available
def test_is_langchain_google_genai_available_true() -> None:
    assert is_langchain_google_genai_available()


@langchain_google_genai_not_available
def test_is_langchain_google_genai_available_false() -> None:
    assert not is_langchain_google_genai_available()


############################
#     langchain_ollama     #
############################


@langchain_ollama_available
def test_check_langchain_ollama_with_package() -> None:
    check_langchain_ollama()


@langchain_ollama_not_available
def test_check_langchain_ollama_without_package() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_ollama' package is required but not installed."
    ):
        check_langchain_ollama()


@langchain_ollama_available
def test_is_langchain_ollama_available_true() -> None:
    assert is_langchain_ollama_available()


@langchain_ollama_not_available
def test_is_langchain_ollama_available_false() -> None:
    assert not is_langchain_ollama_available()


############################
#     langchain_openai     #
############################


@langchain_openai_available
def test_check_langchain_openai_with_package() -> None:
    check_langchain_openai()


@langchain_openai_not_available
def test_check_langchain_openai_without_package() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_openai' package is required but not installed."
    ):
        check_langchain_openai()


@langchain_openai_available
def test_is_langchain_openai_available_true() -> None:
    assert is_langchain_openai_available()


@langchain_openai_not_available
def test_is_langchain_openai_available_false() -> None:
    assert not is_langchain_openai_available()
