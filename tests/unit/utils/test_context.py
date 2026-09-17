from __future__ import annotations

import asyncio
from typing import Any

from zenpyre.utils.context import DelegatingContextManagerMixin


class _FakeStore:
    """A minimal stand-in for BaseRecordStore/BaseDocumentStore."""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.aopened = False
        self.aclosed = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    async def aopen(self) -> None:
        self.aopened = True

    async def aclose(self) -> None:
        self.aclosed = True


class _Wrapper(DelegatingContextManagerMixin):
    _context_managed_attr = "_store"

    def __init__(self, store: Any) -> None:
        self._store = store


#############################################
#     Tests for DelegatingContextManagerMixin     #
#############################################


# --- sync ---


def test_delegating_context_manager_mixin_enter_opens_store() -> None:
    store = _FakeStore()
    with _Wrapper(store):
        assert store.opened


def test_delegating_context_manager_mixin_enter_returns_self() -> None:
    store = _FakeStore()
    wrapper = _Wrapper(store)
    with wrapper as entered:
        assert entered is wrapper


def test_delegating_context_manager_mixin_exit_closes_store() -> None:
    store = _FakeStore()
    with _Wrapper(store):
        pass
    assert store.closed


def test_delegating_context_manager_mixin_exit_closes_store_on_error() -> None:
    store = _FakeStore()
    try:
        with _Wrapper(store):
            msg = "boom"
            raise ValueError(msg)  # noqa: TRY301
    except ValueError:
        pass
    assert store.closed


# --- async ---


def test_delegating_context_manager_mixin_aenter_opens_store() -> None:
    async def run() -> None:
        store = _FakeStore()
        async with _Wrapper(store):
            assert store.aopened

    asyncio.run(run())


def test_delegating_context_manager_mixin_aenter_returns_self() -> None:
    async def run() -> None:
        store = _FakeStore()
        wrapper = _Wrapper(store)
        async with wrapper as entered:
            assert entered is wrapper

    asyncio.run(run())


def test_delegating_context_manager_mixin_aexit_closes_store() -> None:
    async def run() -> _FakeStore:
        store = _FakeStore()
        async with _Wrapper(store):
            pass
        return store

    store = asyncio.run(run())
    assert store.aclosed


def test_delegating_context_manager_mixin_aexit_closes_store_on_error() -> None:
    async def run() -> _FakeStore:
        store = _FakeStore()
        try:
            async with _Wrapper(store):
                msg = "boom"
                raise ValueError(msg)  # noqa: TRY301
        except ValueError:
            pass
        return store

    store = asyncio.run(run())
    assert store.aclosed
