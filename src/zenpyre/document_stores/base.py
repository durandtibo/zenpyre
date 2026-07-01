r"""Provide the abstract base class for document stores."""

from __future__ import annotations

__all__ = ["BaseDocumentStore"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document


class BaseDocumentStore(ABC):
    """Abstract base class for document stores.

    Defines the common interface that all document store implementations
    must provide.  Concrete implementations include
    :class:`~zenpyre.document_stores.DuckDBDocumentStore` (JSON metadata)
    and
    :class:`~zenpyre.document_stores.TypedDuckDBDocumentStore` (typed
    columns + JSON overflow).

    To implement a custom document store, subclass
    :class:`BaseDocumentStore` and implement all abstract methods.
    """

    @abstractmethod
    def add_documents(self, docs: list[Document]) -> None:
        """Add or replace documents in the store.

        Documents whose ``id`` already exists should be replaced
        (upsert semantics).

        Args:
            docs: The list of :class:`~langchain_core.documents.Document`
                instances to add.  Each document must have an ``id``.

        Raises:
            ValueError: If any document has no ``id``.
        """

    @abstractmethod
    def get(self, doc_id: str) -> Document | None:
        """Retrieve a single document by its ID.

        Args:
            doc_id: The document ID to look up.

        Returns:
            The :class:`~langchain_core.documents.Document`, or
            ``None`` if not found.
        """

    @abstractmethod
    def get_many(self, doc_ids: list[str]) -> list[Document | None]:
        """Retrieve multiple documents by their IDs.

        Args:
            doc_ids: The document IDs to look up.

        Returns:
            A list the same length as ``doc_ids``, with the
            corresponding :class:`~langchain_core.documents.Document`
            for each ID that exists, or ``None`` for IDs not found.
        """

    @abstractmethod
    def filter(self, **metadata_filters: Any) -> list[Document]:
        """Retrieve documents matching all provided metadata filters.

        All filters should be combined with ``AND``.  Each keyword
        argument matches the corresponding metadata key exactly.

        Args:
            **metadata_filters: Key-value pairs where each key is a
                metadata field name and the value is the exact value
                to match.  Calling with no arguments should return all
                documents.

        Returns:
            A list of matching
            :class:`~langchain_core.documents.Document` instances.
        """

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Delete a document by its ID.

        IDs that do not exist should be silently ignored.

        Args:
            doc_id: The ID of the document to delete.
        """

    @abstractmethod
    def delete_many(self, doc_ids: list[str]) -> None:
        """Delete multiple documents by their IDs.

        IDs that do not exist should be silently ignored.

        Args:
            doc_ids: The IDs of the documents to delete.
        """

    @abstractmethod
    def all(self) -> list[Document]:
        """Return all documents in the store.

        Returns:
            A list of all :class:`~langchain_core.documents.Document`
            instances currently in the store.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of documents in the store.

        Returns:
            The number of documents currently stored.
        """
