from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from zenpyre.utils.imports import (
    check_langchain_anthropic,
    is_langchain_anthropic_available,
    langchain_anthropic_available,
    raise_langchain_anthropic_missing_error,
)

logger = logging.getLogger(__name__)


MODULE = "zenpyre.utils.imports.langchain_anthropic"


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_langchain_anthropic_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


###############################
#     langchain_anthropic     #
###############################


def test_check_langchain_anthropic_with_package() -> None:
    with patch(f"{MODULE}.is_langchain_anthropic_available", lambda: True):
        check_langchain_anthropic()


def test_check_langchain_anthropic_without_package() -> None:
    with (
        patch(f"{MODULE}.is_langchain_anthropic_available", lambda: False),
        pytest.raises(
            RuntimeError, match=r"'langchain_anthropic' package is required but not installed."
        ),
    ):
        check_langchain_anthropic()


def test_is_langchain_anthropic_available() -> None:
    assert isinstance(is_langchain_anthropic_available(), bool)


def test_langchain_anthropic_available_with_package() -> None:
    with patch(f"{MODULE}.is_langchain_anthropic_available", lambda: True):
        fn = langchain_anthropic_available(my_function)
        assert fn(2) == 44


def test_langchain_anthropic_available_without_package() -> None:
    with patch(f"{MODULE}.is_langchain_anthropic_available", lambda: False):
        fn = langchain_anthropic_available(my_function)
        assert fn(2) is None


def test_langchain_anthropic_available_decorator_with_package() -> None:
    with patch(f"{MODULE}.is_langchain_anthropic_available", lambda: True):

        @langchain_anthropic_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_langchain_anthropic_available_decorator_without_package() -> None:
    with patch(f"{MODULE}.is_langchain_anthropic_available", lambda: False):

        @langchain_anthropic_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_langchain_anthropic_missing_error() -> None:
    with pytest.raises(
        RuntimeError, match=r"'langchain_anthropic' package is required but not installed."
    ):
        raise_langchain_anthropic_missing_error()
