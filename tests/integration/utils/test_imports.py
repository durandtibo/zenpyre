from __future__ import annotations

import pytest

from zenpyre.testing.fixtures import (
    duckdb_not_available,
    langchain_anthropic_available,
    langchain_anthropic_not_available,
    langchain_chroma_available,
    langchain_chroma_not_available,
    langchain_google_genai_available,
    langchain_google_genai_not_available,
    langchain_huggingface_available,
    langchain_huggingface_not_available,
    langchain_ollama_available,
    langchain_ollama_not_available,
    langchain_openai_available,
    langchain_openai_not_available,
    langchain_text_splitters_available,
    langchain_text_splitters_not_available,
)
from zenpyre.utils.imports import (
    check_duckdb,
    check_langchain_anthropic,
    check_langchain_chroma,
    check_langchain_google_genai,
    check_langchain_huggingface,
    check_langchain_ollama,
    check_langchain_openai,
    check_langchain_text_splitters,
    duckdb_available,
    is_duckdb_available,
    is_langchain_anthropic_available,
    is_langchain_chroma_available,
    is_langchain_google_genai_available,
    is_langchain_huggingface_available,
    is_langchain_ollama_available,
    is_langchain_openai_available,
    is_langchain_text_splitters_available,
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


############################
#     langchain_chroma     #
############################


@langchain_chroma_available
def test_check_langchain_chroma_with_package() -> None:
    check_langchain_chroma()


@langchain_chroma_not_available
def test_check_langchain_chroma_without_package() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_chroma' package is required but not installed."
    ):
        check_langchain_chroma()


@langchain_chroma_available
def test_is_langchain_chroma_available_true() -> None:
    assert is_langchain_chroma_available()


@langchain_chroma_not_available
def test_is_langchain_chroma_available_false() -> None:
    assert not is_langchain_chroma_available()


##################
#     duckdb     #
##################


@duckdb_available
def test_check_duckdb_with_package() -> None:
    check_duckdb()


@duckdb_not_available
def test_check_duckdb_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'duckdb' package is required but not installed."):
        check_duckdb()


@duckdb_available
def test_is_duckdb_available_true() -> None:
    assert is_duckdb_available()


@duckdb_not_available
def test_is_duckdb_available_false() -> None:
    assert not is_duckdb_available()


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


#################################
#     langchain_huggingface     #
#################################


@langchain_huggingface_available
def test_check_langchain_huggingface_with_package() -> None:
    check_langchain_huggingface()


@langchain_huggingface_not_available
def test_check_langchain_huggingface_without_package() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_huggingface' package is required but not installed."
    ):
        check_langchain_huggingface()


@langchain_huggingface_available
def test_is_langchain_huggingface_available_true() -> None:
    assert is_langchain_huggingface_available()


@langchain_huggingface_not_available
def test_is_langchain_huggingface_available_false() -> None:
    assert not is_langchain_huggingface_available()


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


####################################
#     langchain_text_splitters     #
####################################


@langchain_text_splitters_available
def test_check_langchain_text_splitters_with_package() -> None:
    check_langchain_text_splitters()


@langchain_text_splitters_not_available
def test_check_langchain_text_splitters_without_package() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_text_splitters' package is required but not installed."
    ):
        check_langchain_text_splitters()


@langchain_text_splitters_available
def test_is_langchain_text_splitters_available_true() -> None:
    assert is_langchain_text_splitters_available()


@langchain_text_splitters_not_available
def test_is_langchain_text_splitters_available_false() -> None:
    assert not is_langchain_text_splitters_available()
