from __future__ import annotations

from unittest.mock import Mock

import pytest
from coola.hashing import HasherRegistry, get_default_registry
from langchain_core.load import Serializable
from langchain_core.messages import AIMessage, HumanMessage

from zenpyre.runnables.hashing import SerializableHasher


@pytest.fixture
def registry() -> HasherRegistry:
    return get_default_registry()


########################################
#     Tests for SerializableHasher     #
########################################


def test_serializable_hasher_repr() -> None:
    assert repr(SerializableHasher()) == "SerializableHasher()"


def test_serializable_hasher_str() -> None:
    assert str(SerializableHasher()) == "SerializableHasher()"


def test_serializable_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    hasher = SerializableHasher()
    message = HumanMessage(content="hello")
    assert isinstance(hasher.hash(message, registry=registry), str)


def test_serializable_hasher_hash_default_length(registry: HasherRegistry) -> None:
    hasher = SerializableHasher()
    message = HumanMessage(content="hello")
    assert len(hasher.hash(message, registry=registry)) == 64


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(2, id="min-valid"),
        pytest.param(32, id="middle"),
        pytest.param(64, id="default"),
        pytest.param(128, id="max-valid"),
    ],
)
def test_serializable_hasher_hash_length(registry: HasherRegistry, length: int) -> None:
    hasher = SerializableHasher()
    message = HumanMessage(content="hello")
    assert len(hasher.hash(message, registry=registry, length=length)) == length


def test_serializable_hasher_hash_same_object_same_hash(registry: HasherRegistry) -> None:
    hasher = SerializableHasher()
    message = HumanMessage(content="hello")
    assert hasher.hash(message, registry=registry) == hasher.hash(message, registry=registry)


def test_serializable_hasher_hash_equal_objects_same_hash(registry: HasherRegistry) -> None:
    hasher = SerializableHasher()
    message_a = HumanMessage(content="hello")
    message_b = HumanMessage(content="hello")
    assert hasher.hash(message_a, registry=registry) == hasher.hash(message_b, registry=registry)


def test_serializable_hasher_hash_different_content_different_hash(
    registry: HasherRegistry,
) -> None:
    hasher = SerializableHasher()
    message_a = HumanMessage(content="hello")
    message_b = HumanMessage(content="world")
    assert hasher.hash(message_a, registry=registry) != hasher.hash(message_b, registry=registry)


def test_serializable_hasher_hash_different_type_different_hash(registry: HasherRegistry) -> None:
    # Same content, different Serializable subclass -- to_json() includes
    # the class id, so these must not collide.
    hasher = SerializableHasher()
    human = HumanMessage(content="hello")
    ai = AIMessage(content="hello")
    assert hasher.hash(human, registry=registry) != hasher.hash(ai, registry=registry)


def test_serializable_hasher_hash_matches_registry_hash_of_to_json(
    registry: HasherRegistry,
) -> None:
    hasher = SerializableHasher()
    message = HumanMessage(content="hello")
    assert hasher.hash(message, registry=registry) == registry.hash(message.to_json())


def test_serializable_hasher_hash_matches_registry_hash_with_length(
    registry: HasherRegistry,
) -> None:
    hasher = SerializableHasher()
    message = HumanMessage(content="hello")
    assert hasher.hash(message, registry=registry, length=32) == registry.hash(
        message.to_json(), length=32
    )


def test_serializable_hasher_hash_delegates_to_given_registry() -> None:
    # Verifies the registry argument is actually used (not ignored in
    # favor of a fixed global registry).
    hasher = SerializableHasher()
    message = HumanMessage(content="hello")
    mock_registry = Mock(spec=HasherRegistry)
    mock_registry.hash.return_value = "mock-hash"

    result = hasher.hash(message, registry=mock_registry, length=32)

    assert result == "mock-hash"
    mock_registry.hash.assert_called_once_with(message.to_json(), length=32)


def test_serializable_hasher_hash_uses_registry_not_a_different_one() -> None:
    # Two different mock registries given the same input must each be
    # queried independently -- confirms there's no fallback to some
    # other fixed/global registry under the hood.
    hasher = SerializableHasher()
    message = HumanMessage(content="hello")
    mock_registry_a = Mock(spec=HasherRegistry)
    mock_registry_a.hash.return_value = "hash-a"
    mock_registry_b = Mock(spec=HasherRegistry)
    mock_registry_b.hash.return_value = "hash-b"

    hasher.hash(message, registry=mock_registry_a)
    hasher.hash(message, registry=mock_registry_b)

    mock_registry_a.hash.assert_called_once()
    mock_registry_b.hash.assert_called_once()


def test_serializable_hasher_hash_raises_type_error_for_non_serializable(
    registry: HasherRegistry,
) -> None:
    hasher = SerializableHasher()
    obj = Serializable()
    with pytest.raises(TypeError, match=r"Cannot hash non-serializable object of type"):
        hasher.hash(obj, registry=registry)


def test_serializable_hasher_hash_raises_type_error_includes_type_name(
    registry: HasherRegistry,
) -> None:
    hasher = SerializableHasher()
    obj = Serializable()
    with pytest.raises(TypeError, match=r"Serializable"):
        hasher.hash(obj, registry=registry)


def test_serializable_hasher_hash_raises_before_touching_registry() -> None:
    # The is_lc_serializable() check must happen before any registry
    # lookup, so a non-serializable object never reaches registry.hash
    # at all (relevant since to_json() would return a meaningless
    # "not implemented" sentinel for it).
    hasher = SerializableHasher()
    obj = Serializable()
    mock_registry = Mock(spec=HasherRegistry)

    with pytest.raises(TypeError):
        hasher.hash(obj, registry=mock_registry)

    mock_registry.hash.assert_not_called()
