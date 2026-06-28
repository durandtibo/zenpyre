from __future__ import annotations

from types import ModuleType

import pytest

from zenpyre.utils.fallback.langchain_huggingface import langchain_huggingface


def test_langchain_huggingface_is_module_type() -> None:
    assert isinstance(langchain_huggingface, ModuleType)


def test_langchain_huggingface_module_name() -> None:
    assert langchain_huggingface.__name__ == "langchain_huggingface"


def test_langchain_huggingface_array_class_exists() -> None:
    assert hasattr(langchain_huggingface, "Array")


def test_langchain_huggingface_array_is_class() -> None:
    assert isinstance(langchain_huggingface.Array, type)


def test_langchain_huggingface_array_instantiation() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_huggingface' package is required but not installed."
    ):
        langchain_huggingface.HuggingFaceEmbeddings()


def test_langchain_huggingface_array_instantiation_with_args() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_huggingface' package is required but not installed."
    ):
        langchain_huggingface.HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
