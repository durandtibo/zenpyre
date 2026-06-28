r"""Provide a concrete default factory for LangChain embedding
models."""

from __future__ import annotations

__all__ = ["EmbeddingsFactory"]

from typing import TYPE_CHECKING, Any

from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

from zenpyre.embeddings.factory.base import BaseEmbeddingsFactory

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class EmbeddingsFactory(BaseEmbeddingsFactory):
    """A concrete embedding factory that wraps a pre-built
    :class:`~langchain_core.embeddings.Embeddings` instance.

    Use this when the embedding model is already instantiated and you
    simply want to wrap it in the :class:`~BaseEmbeddingsFactory`
    interface — for example, when injecting a fixed embedding model
    into a component that expects a factory.

    Args:
        embeddings: A fully configured
            :class:`~langchain_core.embeddings.Embeddings` instance to
            return from :meth:`make_embeddings`.

    Example:
        ```pycon
        >>> from langchain_ollama import OllamaEmbeddings
        >>> from zenpyre.embeddings.factory import EmbeddingsFactory
        >>> factory = EmbeddingsFactory(OllamaEmbeddings(model="nomic-embed-text"))
        >>> embeddings = factory.make_embeddings()

        ```
    """

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self._get_repr_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._get_repr_kwargs()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def make_embeddings(self) -> Embeddings:
        return self._embeddings

    def _get_repr_kwargs(self) -> dict[str, Any]:
        """Return a display-friendly dict of constructor arguments.

        Returns:
            A dict with the ``"embeddings"`` key.
        """
        return {"embeddings": self._embeddings}
