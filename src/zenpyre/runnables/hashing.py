r"""Provide hashing utilities."""

from __future__ import annotations

__all__ = ["SerializableHasher"]


from coola.hashing import BaseHasher, HasherRegistry, get_default_registry
from langchain_core.load import Serializable


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
