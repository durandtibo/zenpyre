from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from zenpyre.utils.imports import (
    check_langchain_chroma,
    is_langchain_chroma_available,
    langchain_chroma_available,
    raise_langchain_chroma_missing_error,
)

logger = logging.getLogger(__name__)


MODULE = "zenpyre.utils.imports.langchain_chroma"


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_langchain_chroma_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


############################
#     langchain_chroma     #
############################


def test_check_langchain_chroma_with_package() -> None:
    with patch(f"{MODULE}.is_langchain_chroma_available", lambda: True):
        check_langchain_chroma()


def test_check_langchain_chroma_without_package() -> None:
    with (
        patch(f"{MODULE}.is_langchain_chroma_available", lambda: False),
        pytest.raises(
            RuntimeError, match=r"'langchain_chroma' package is required but not installed."
        ),
    ):
        check_langchain_chroma()


def test_is_langchain_chroma_available() -> None:
    assert isinstance(is_langchain_chroma_available(), bool)


def test_langchain_chroma_available_with_package() -> None:
    with patch(f"{MODULE}.is_langchain_chroma_available", lambda: True):
        fn = langchain_chroma_available(my_function)
        assert fn(2) == 44


def test_langchain_chroma_available_without_package() -> None:
    with patch(f"{MODULE}.is_langchain_chroma_available", lambda: False):
        fn = langchain_chroma_available(my_function)
        assert fn(2) is None


def test_langchain_chroma_available_decorator_with_package() -> None:
    with patch(f"{MODULE}.is_langchain_chroma_available", lambda: True):

        @langchain_chroma_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_langchain_chroma_available_decorator_without_package() -> None:
    with patch(f"{MODULE}.is_langchain_chroma_available", lambda: False):

        @langchain_chroma_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_langchain_chroma_missing_error() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_chroma' package is required but not installed."
    ):
        raise_langchain_chroma_missing_error()
