from __future__ import annotations

from types import ModuleType

import pytest

from zenpyre.utils.fallback.langchain_text_splitters import langchain_text_splitters


def test_langchain_text_splitters_is_module_type() -> None:
    assert isinstance(langchain_text_splitters, ModuleType)


def test_langchain_text_splitters_module_name() -> None:
    assert langchain_text_splitters.__name__ == "langchain_text_splitters"


def test_langchain_text_splitters_embeddings_class_exists() -> None:
    assert hasattr(langchain_text_splitters, "TextSplitter")


def test_langchain_text_splitters_embeddings_is_class() -> None:
    assert isinstance(langchain_text_splitters.TextSplitter, type)


def test_langchain_text_splitters_array_instantiation() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_text_splitters' package is required but not installed."
    ):
        langchain_text_splitters.TextSplitter()


def test_langchain_text_splitters_array_instantiation_with_args() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_text_splitters' package is required but not installed."
    ):
        langchain_text_splitters.TextSplitter(model_name="all-MiniLM-L6-v2")
