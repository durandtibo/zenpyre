r"""Contain embedding factories."""

from __future__ import annotations

__all__ = ["BaseEmbeddingsFactory", "EmbeddingsFactory"]

from zenpyre.embeddings.factory.base import BaseEmbeddingsFactory
from zenpyre.embeddings.factory.vanilla import EmbeddingsFactory
