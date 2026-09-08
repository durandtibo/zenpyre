r"""Provide a document loader backed by a document store."""

from __future__ import annotations

__all__ = ["DocumentStoreLoader"]


from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from langchain_core.document_loaders import BaseLoader

if TYPE_CHECKING:
    from collections.abc import Iterator

    from docculus.store.base import BaseDocumentStore
    from langchain_core.documents import Document


class DocumentStoreLoader(BaseLoader, MultilineDisplayMixin):
    """A loader that yields documents from a :class:`BaseDocumentStore`.

    Use this when documents already live in a document store and you
    need to wrap them in a
    :class:`~langchain_core.document_loaders.BaseLoader` interface —
    for example, to feed a store's contents into a pipeline (e.g. a
    vector store indexer) that expects a loader.

    Each call to :meth:`lazy_load` (and therefore :meth:`load`) uses
    ``store`` as a context manager, so it's opened before reading and
    closed once the read completes (or is abandoned), rather than
    relying on the caller to open/close it. ``open``/``close`` are
    expected to be idempotent (per :class:`BaseDocumentStore`'s
    contract), so ``store`` may still be reused across multiple loads
    or shared with other code.

    Args:
        store: The :class:`~docculus.store.base.BaseDocumentStore`
            to load documents from.

    Example:
        ```pycon
        >>> from langchain_core.documents import Document
        >>> from zenpyre.document_loaders import DocumentStoreLoader
        >>> from docculus.store import InMemoryDocumentStore
        >>> store = InMemoryDocumentStore()
        >>> store.open()
        >>> store.set_many(
        ...     [Document(id="1", page_content="Hello"), Document(id="2", page_content="World")]
        ... )
        >>> loader = DocumentStoreLoader(store)
        >>> docs = loader.load()

        ```
    """

    def __init__(self, store: BaseDocumentStore) -> None:
        self._store = store

    def lazy_load(self) -> Iterator[Document]:
        with self._store:
            yield from self._store.values()

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"store": self._store}
