"""Unit tests for HuggingFaceEmbeddingsFactory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings

from zenpyre.embeddings.factory import (
    BaseEmbeddingsFactory,
    HuggingFaceEmbeddingsFactory,
)

MODULE = "zenpyre.embeddings.factory.huggingface"


def _make_mock_embeddings() -> MagicMock:
    return MagicMock(spec=Embeddings)


##################################################
#     Tests for HuggingFaceEmbeddingsFactory     #
##################################################


# --- Inheritance ---


def test_huggingface_embeddings_factory_is_base_embeddings_factory() -> None:
    assert isinstance(HuggingFaceEmbeddingsFactory(), BaseEmbeddingsFactory)


# --- make_embeddings ---


def test_huggingface_embeddings_factory_make_embeddings_returns_embeddings() -> None:
    mock = _make_mock_embeddings()
    with patch(f"{MODULE}.HuggingFaceEmbeddings", return_value=mock):
        result = HuggingFaceEmbeddingsFactory().make_embeddings()
    assert isinstance(result, Embeddings)


def test_huggingface_embeddings_factory_make_embeddings_forwards_kwargs() -> None:
    with patch(f"{MODULE}.HuggingFaceEmbeddings") as mock_cls:
        HuggingFaceEmbeddingsFactory(model_name="all-MiniLM-L6-v2").make_embeddings()
    mock_cls.assert_called_once_with(model_name="all-MiniLM-L6-v2")


def test_huggingface_embeddings_factory_make_embeddings_no_kwargs() -> None:
    with patch(f"{MODULE}.HuggingFaceEmbeddings") as mock_cls:
        HuggingFaceEmbeddingsFactory().make_embeddings()
    mock_cls.assert_called_once_with()


def test_huggingface_embeddings_factory_make_embeddings_multiple_kwargs() -> None:
    with patch(f"{MODULE}.HuggingFaceEmbeddings") as mock_cls:
        HuggingFaceEmbeddingsFactory(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        ).make_embeddings()
    mock_cls.assert_called_once_with(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def test_huggingface_embeddings_factory_make_embeddings_each_call_creates_new_instance() -> None:
    factory = HuggingFaceEmbeddingsFactory(model_name="all-MiniLM-L6-v2")
    with patch(f"{MODULE}.HuggingFaceEmbeddings") as mock_cls:
        mock_cls.side_effect = [_make_mock_embeddings(), _make_mock_embeddings()]
        first = factory.make_embeddings()
        second = factory.make_embeddings()
    assert first is not second


# --- _get_repr_kwargs ---


def test_huggingface_embeddings_factory_get_repr_kwargs_empty() -> None:
    assert HuggingFaceEmbeddingsFactory()._get_repr_kwargs() == {}


def test_huggingface_embeddings_factory_get_repr_kwargs_with_model_name() -> None:
    factory = HuggingFaceEmbeddingsFactory(model_name="all-MiniLM-L6-v2")
    assert factory._get_repr_kwargs() == {"model_name": "all-MiniLM-L6-v2"}


def test_huggingface_embeddings_factory_get_repr_kwargs_with_multiple_kwargs() -> None:
    factory = HuggingFaceEmbeddingsFactory(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    assert factory._get_repr_kwargs() == {
        "model_name": "all-MiniLM-L6-v2",
        "model_kwargs": {"device": "cpu"},
    }


# --- __repr__ and __str__ ---


def test_huggingface_embeddings_factory_repr_no_kwargs() -> None:
    assert repr(HuggingFaceEmbeddingsFactory()) == "HuggingFaceEmbeddingsFactory()"


def test_huggingface_embeddings_factory_str_no_kwargs() -> None:
    assert str(HuggingFaceEmbeddingsFactory()) == "HuggingFaceEmbeddingsFactory()"


def test_huggingface_embeddings_factory_repr_with_model_name() -> None:
    factory = HuggingFaceEmbeddingsFactory(model_name="all-MiniLM-L6-v2")
    assert repr(factory) == "HuggingFaceEmbeddingsFactory(model_name='all-MiniLM-L6-v2')"


def test_huggingface_embeddings_factory_str_with_model_name() -> None:
    factory = HuggingFaceEmbeddingsFactory(model_name="all-MiniLM-L6-v2")
    assert str(factory) == "HuggingFaceEmbeddingsFactory(model_name=all-MiniLM-L6-v2)"
