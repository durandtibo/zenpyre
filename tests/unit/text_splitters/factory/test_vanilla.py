from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.testing.fixtures import langchain_text_splitters_available
from zenpyre.text_splitters.factory import BaseTextSplitterFactory, TextSplitterFactory
from zenpyre.utils.imports import is_langchain_text_splitters_available

if is_langchain_text_splitters_available():
    from langchain_text_splitters import CharacterTextSplitter, TextSplitter


#########################################
#     Tests for TextSplitterFactory     #
#########################################


# --- Inheritance ---


@langchain_text_splitters_available
def test_text_splitter_factory_is_base_text_splitter_factory() -> None:
    assert isinstance(TextSplitterFactory(CharacterTextSplitter()), BaseTextSplitterFactory)


# --- make_text_splitter ---


@langchain_text_splitters_available
def test_text_splitter_factory_make_text_splitter_returns_text_splitter() -> None:
    factory = TextSplitterFactory(CharacterTextSplitter())
    assert isinstance(factory.make_text_splitter(), TextSplitter)


@langchain_text_splitters_available
def test_text_splitter_factory_make_text_splitter_returns_same_instance() -> None:
    splitter = CharacterTextSplitter()
    factory = TextSplitterFactory(splitter)
    assert factory.make_text_splitter() is splitter


@langchain_text_splitters_available
def test_text_splitter_factory_make_text_splitter_returns_same_instance_on_repeated_calls() -> None:
    factory = TextSplitterFactory(CharacterTextSplitter())
    assert factory.make_text_splitter() is factory.make_text_splitter()


@langchain_text_splitters_available
def test_text_splitter_factory_different_instances_independent() -> None:
    factory_a = TextSplitterFactory(CharacterTextSplitter())
    factory_b = TextSplitterFactory(CharacterTextSplitter())
    assert factory_a.make_text_splitter() is not factory_b.make_text_splitter()


# --- _get_repr_kwargs ---


@langchain_text_splitters_available
def test_text_splitter_factory_get_repr_kwargs() -> None:
    splitter = CharacterTextSplitter()
    factory = TextSplitterFactory(splitter)
    assert objects_are_equal(factory._get_repr_kwargs(), {"text_splitter": splitter})


# --- __repr__ and __str__ ---


@langchain_text_splitters_available
def test_text_splitter_factory_repr_contains_class_name() -> None:
    factory = TextSplitterFactory(CharacterTextSplitter())
    assert repr(factory).startswith("TextSplitterFactory(")


@langchain_text_splitters_available
def test_text_splitter_factory_str_contains_class_name() -> None:
    factory = TextSplitterFactory(CharacterTextSplitter())
    assert str(factory).startswith("TextSplitterFactory(")


@langchain_text_splitters_available
def test_text_splitter_factory_repr_contains_text_splitter() -> None:
    factory = TextSplitterFactory(CharacterTextSplitter())
    assert "text_splitter" in repr(factory)


@langchain_text_splitters_available
def test_text_splitter_factory_repr_equals_str() -> None:
    factory = TextSplitterFactory(CharacterTextSplitter())
    assert repr(factory) == str(factory)
