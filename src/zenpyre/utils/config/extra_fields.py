"""Contain configurations."""

from __future__ import annotations

__all__ = ["ExtraFieldsConfig"]

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from zenpyre.utils.config.base import BaseConfig


@dataclass(frozen=True, kw_only=True)
class ExtraFieldsConfig(BaseConfig):
    """Base for configs that merge arbitrary keyword arguments ("extra")
    into :meth:`to_kwargs`, alongside whatever typed fields a subclass
    declares.

    Subclasses just need to be frozen dataclasses with their own typed
    fields; :meth:`to_kwargs`, the extra/field-name collision check, and
    ``__hash__`` are all inherited from here and work automatically via
    introspection (:func:`dataclasses.fields`) — no per-subclass
    overriding required.

    ``extra`` is declared keyword-only (``kw_only=True`` on this class)
    specifically so that subclasses are free to add non-defaulted,
    positional-or-keyword fields of their own without hitting
    dataclass's "non-default argument follows default argument" error,
    which would otherwise apply because ``extra`` (with its default)
    is defined here in the base class.

    Attributes:
        extra: Additional keyword arguments merged into
            :meth:`to_kwargs`. Must not contain a key that collides
            with any of this config's own field names (including ones
            declared by a subclass).
    """

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reserved = {f.name for f in dataclasses.fields(self) if f.name != "extra"}
        collisions = reserved & self.extra.keys()
        if collisions:
            msg = (
                f"'extra' must not contain any of this config's own field names "
                f"{sorted(reserved)}; got overlapping key(s) {sorted(collisions)} "
                f"in extra={dict(self.extra)!r}."
            )
            raise ValueError(msg)
        # Wrap `extra` in a read-only view so mutating it after
        # construction raises, matching the `frozen=True` contract
        # (which otherwise only prevents reassigning the attribute
        # itself, not mutating the dict it points to).
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def to_kwargs(self) -> dict[str, Any]:
        """Return every dataclass field (including ones declared by a
        subclass) merged with ``extra``, as a flat dict.

        Fields are collected via introspection
        (:func:`dataclasses.fields`) rather than hardcoded by name, so
        a subclass that adds a new typed field gets it included here
        automatically, with no need to override this method.

        Returns:
            A dict of keyword arguments representing this
                configuration.
        """
        kwargs = {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "extra"
        }
        kwargs.update(self.extra)
        return kwargs

    def __hash__(self) -> int:
        # The auto-generated dataclass __hash__ would hash `extra`
        # (and any nested BaseConfig field, unless it's independently
        # hashable) directly, which fails for an unhashable
        # dict/MappingProxyType. Hashing the (string) cache key
        # sidesteps that while keeping equal configs hash-equal.
        return hash(self.cache_key())
