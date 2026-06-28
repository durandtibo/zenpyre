from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from zenpyre.utils.imports import (
    check_langchain_huggingface,
    is_langchain_huggingface_available,
    langchain_huggingface_available,
    raise_langchain_huggingface_missing_error,
)

logger = logging.getLogger(__name__)


MODULE = "zenpyre.utils.imports.langchain_huggingface"


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_langchain_huggingface_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


#################################
#     langchain_huggingface     #
#################################


def test_check_langchain_huggingface_with_package() -> None:
    with patch(f"{MODULE}.is_langchain_huggingface_available", lambda: True):
        check_langchain_huggingface()


def test_check_langchain_huggingface_without_package() -> None:
    with (
        patch(f"{MODULE}.is_langchain_huggingface_available", lambda: False),
        pytest.raises(
            RuntimeError, match=r"'langchain_huggingface' package is required but not installed."
        ),
    ):
        check_langchain_huggingface()


def test_is_langchain_huggingface_available() -> None:
    assert isinstance(is_langchain_huggingface_available(), bool)


def test_langchain_huggingface_available_with_package() -> None:
    with patch(f"{MODULE}.is_langchain_huggingface_available", lambda: True):
        fn = langchain_huggingface_available(my_function)
        assert fn(2) == 44


def test_langchain_huggingface_available_without_package() -> None:
    with patch(f"{MODULE}.is_langchain_huggingface_available", lambda: False):
        fn = langchain_huggingface_available(my_function)
        assert fn(2) is None


def test_langchain_huggingface_available_decorator_with_package() -> None:
    with patch(f"{MODULE}.is_langchain_huggingface_available", lambda: True):

        @langchain_huggingface_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_langchain_huggingface_available_decorator_without_package() -> None:
    with patch(f"{MODULE}.is_langchain_huggingface_available", lambda: False):

        @langchain_huggingface_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_langchain_huggingface_missing_error() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_huggingface' package is required but not installed."
    ):
        raise_langchain_huggingface_missing_error()
