r"""Provide UUID hashing utilities for Python dictionaries."""

from __future__ import annotations

__all__ = ["SerializableHasher", "hash_dict_uuid"]

import json
import uuid
from typing import Any

from coola.hashing import BaseHasher, HasherRegistry, get_default_registry
from langchain_core.load import Serializable

# Project-specific namespace for deterministic dict UUIDs.
# Generated once with uuid.uuid4() and fixed here so hashes are
# stable across runs and reproducible across environments.
_NAMESPACE = uuid.UUID("21e6c43e-bc36-4f09-8e20-98201adab5df")


def hash_dict_uuid(data: dict[str, Any]) -> str:
    """Compute a stable, reproducible UUID for a Python dictionary.

    Serialises ``data`` via :func:`json.dumps` with ``sort_keys=True``
    to guarantee a consistent ordering regardless of dict insertion
    order, then derives a deterministic UUID using :func:`uuid.uuid5`
    with a fixed project-specific namespace.

    Args:
        data: The dictionary to hash.  All values must be
            JSON-serialisable.

    Returns:
        A lowercase UUID string of the form
        ``'xxxxxxxx-xxxx-5xxx-xxxx-xxxxxxxxxxxx'``.

    Raises:
        TypeError: If any value in ``data`` is not JSON-serialisable.

    Example:
        ```pycon
        >>> from zenpyre.utils.hashing import hash_dict_uuid
        >>> hash_dict_uuid({"source": "cats.txt", "page": 1})  # doctest: +ELLIPSIS
        '...'
        >>> hash_dict_uuid({"page": 1, "source": "cats.txt"}) == hash_dict_uuid(
        ...     {"source": "cats.txt", "page": 1}
        ... )
        True

        ```
    """
    return str(uuid.uuid5(_NAMESPACE, json.dumps(data, sort_keys=True)))


class SerializableHasher(BaseHasher[Serializable]):
    r"""Hasher for LangChain ``Serializable`` objects.

    This hasher applies ``registry.hash`` to the object's ``to_json()``
    output (its LangChain-serialized form: class id plus constructor
    kwargs), so two ``Serializable`` instances with equal serialized
    content produce the same hash regardless of object identity. Using
    the supplied ``registry`` (rather than a fixed global one) means
    nested values within ``to_json()`` are hashed with whatever hashers
    ``registry`` has registered, so ``registry`` must have hashers
    registered for the primitive types ``to_json()`` can produce (at
    minimum ``dict``, ``list``, and ``str``) -- ``coola``'s
    ``get_default_registry()`` satisfies this out of the box; a bare,
    empty ``HasherRegistry()`` does not.

    Raises:
        TypeError: Raised by :meth:`hash` if ``data`` is not
            LangChain-serializable (``data.is_lc_serializable()`` is
            ``False``). In that case ``to_json()`` would return a
            generic "not implemented" sentinel shared by every
            non-serializable instance, which would silently hash all
            such instances to the same value regardless of their actual
            content -- raising instead of doing that.

    Example:
        ```pycon
        >>> from langchain_core.messages import HumanMessage
        >>> from coola.hashing import get_default_registry
        >>> from zenpyre.utils.hashing import SerializableHasher
        >>> registry = get_default_registry()
        >>> hasher = SerializableHasher()
        >>> hasher
        SerializableHasher()
        >>> message = HumanMessage(content="hello")
        >>> len(hasher.hash(message, registry=registry))
        64

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def hash(
        self,
        data: Serializable,
        registry: HasherRegistry,
        length: int = 64,
    ) -> str:
        if not data.is_lc_serializable():
            msg = f"Cannot hash non-serializable object of type {type(data).__qualname__}"
            raise TypeError(msg)
        return registry.hash(data.to_json(), length=length)


get_default_registry().register(Serializable, SerializableHasher(), exist_ok=True)
