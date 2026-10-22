r"""Provide a mixin to delegate the context-manager protocol to a named
inner attribute."""

from __future__ import annotations

__all__ = ["DelegatingContextManagerMixin"]

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Self


class DelegatingContextManagerMixin:
    """Provide sync and async context-manager support by delegating to
    a named inner attribute's ``open``/``close``/``aopen``/``aclose``
    methods.

    A subclass sets :attr:`_context_managed_attr` to the name of an
    instance attribute exposing ``open()``, ``close()``, ``aopen()``,
    and ``aclose()`` (e.g. a
    :class:`~persista.record.store.base.BaseRecordStore` or a
    :class:`~docculus.store.base.BaseDocumentStore`). This mixin then
    provides ``__enter__``/``__exit__``/``__aenter__``/``__aexit__``
    that open that attribute on entry and close it on exit (regardless
    of whether the block raised), for both ``with`` and ``async with``.

    Example:
        ```pycon
        >>> from zenpyre.utils.context import DelegatingContextManagerMixin
        >>> class Wrapper(DelegatingContextManagerMixin):
        ...     _context_managed_attr = "_store"
        ...     def __init__(self, store):
        ...         self._store = store
        ...

        ```
    """

    _context_managed_attr: ClassVar[str]

    def __enter__(self) -> Self:
        """Open the delegate attribute and return this object.

        Returns:
            This object, unchanged.
        """
        getattr(self, self._context_managed_attr).open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the delegate attribute, regardless of whether the
        ``with`` block raised.

        Args:
            exc_type: The exception type, if the ``with`` block
                raised; otherwise ``None``.
            exc_value: The exception instance, if the ``with`` block
                raised; otherwise ``None``.
            traceback: The exception's traceback, if the ``with``
                block raised; otherwise ``None``.
        """
        getattr(self, self._context_managed_attr).close()

    async def __aenter__(self) -> Self:
        """Asynchronously open the delegate attribute and return this
        object.

        Returns:
            This object, unchanged.
        """
        await getattr(self, self._context_managed_attr).aopen()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Asynchronously close the delegate attribute, regardless of
        whether the ``async with`` block raised.

        Args:
            exc_type: The exception type, if the ``async with`` block
                raised; otherwise ``None``.
            exc_value: The exception instance, if the ``async with``
                block raised; otherwise ``None``.
            traceback: The exception's traceback, if the ``async
                with`` block raised; otherwise ``None``.
        """
        await getattr(self, self._context_managed_attr).aclose()
