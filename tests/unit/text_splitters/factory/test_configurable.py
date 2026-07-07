from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.testing.fixtures import langchain_text_splitters_available
from zenpyre.text_splitters.factory import (
    BaseTextSplitterFactory,
    ConfigurableTextSplitterFactory,
)
from zenpyre.utils.imports import is_langchain_text_splitters_available

if is_langchain_text_splitters_available():
    from langchain_text_splitters import CharacterTextSplitter, TextSplitter

CHARACTER_TEXT_SPLITTER_TARGET = "langchain_text_splitters.CharacterTextSplitter"


###################################################
#     Tests for ConfigurableTextSplitterFactory   #
###################################################


# --- Inheritance ---


@langchain_text_splitters_available
def test_configurable_text_splitter_factory_is_base_text_splitter_factory() -> None:
    assert isinstance(
        ConfigurableTextSplitterFactory(CharacterTextSplitter()), BaseTextSplitterFactory
    )


# --- make_text_splitter from instance ---


@langchain_text_splitters_available
def test_configurable_text_splitter_factory_make_text_splitter_returns_text_splitter() -> None:
    factory = ConfigurableTextSplitterFactory(CharacterTextSplitter())
    assert isinstance(factory.make_text_splitter(), TextSplitter)


@langchain_text_splitters_available
def test_configurable_text_splitter_factory_make_text_splitter_returns_same_instance() -> None:
    splitter = CharacterTextSplitter()
    factory = ConfigurableTextSplitterFactory(splitter)
    assert factory.make_text_splitter() is splitter


# --- make_text_splitter from dict ---


@langchain_text_splitters_available
def test_configurable_text_splitter_factory_make_text_splitter_from_dict_returns_text_splitter() -> (
    None
):
    factory = ConfigurableTextSplitterFactory({"_target_": CHARACTER_TEXT_SPLITTER_TARGET})
    assert isinstance(factory.make_text_splitter(), TextSplitter)


@langchain_text_splitters_available
def test_configurable_text_splitter_factory_make_text_splitter_from_dict_returns_correct_type() -> (
    None
):
    factory = ConfigurableTextSplitterFactory({"_target_": CHARACTER_TEXT_SPLITTER_TARGET})
    assert isinstance(factory.make_text_splitter(), CharacterTextSplitter)


# --- _get_repr_kwargs ---


@langchain_text_splitters_available
def test_configurable_text_splitter_factory_get_repr_kwargs_instance() -> None:
    splitter = CharacterTextSplitter()
    factory = ConfigurableTextSplitterFactory(splitter)
    assert objects_are_equal(factory._get_repr_kwargs(), {"text_splitter": splitter})


@langchain_text_splitters_available
def test_configurable_text_splitter_factory_get_repr_kwargs_dict_input() -> None:
    config = {"_target_": CHARACTER_TEXT_SPLITTER_TARGET}
    factory = ConfigurableTextSplitterFactory(config)
    assert objects_are_equal(factory._get_repr_kwargs(), {"text_splitter": config})


# --- __repr__ and __str__ ---


@langchain_text_splitters_available
def test_configurable_text_splitter_factory_repr_starts_with_class_name() -> None:
    factory = ConfigurableTextSplitterFactory(CharacterTextSplitter())
    assert repr(factory).startswith("ConfigurableTextSplitterFactory(")


@langchain_text_splitters_available
def test_configurable_text_splitter_factory_str_starts_with_class_name() -> None:
    factory = ConfigurableTextSplitterFactory(CharacterTextSplitter())
    assert str(factory).startswith("ConfigurableTextSplitterFactory(")


@langchain_text_splitters_available
def test_configurable_text_splitter_factory_repr_contains_text_splitter() -> None:
    factory = ConfigurableTextSplitterFactory(CharacterTextSplitter())
    assert "text_splitter" in repr(factory)


@langchain_text_splitters_available
def test_configurable_text_splitter_factory_repr_equals_str() -> None:
    factory = ConfigurableTextSplitterFactory(CharacterTextSplitter())
    assert repr(factory) == str(factory)
