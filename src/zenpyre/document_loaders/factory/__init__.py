r"""Contain factories for document loaders."""

from __future__ import annotations

__all__ = ["BaseLoaderFactory", "LoaderFactory"]

from zenpyre.document_loaders.factory.base import BaseLoaderFactory
from zenpyre.document_loaders.factory.vanilla import LoaderFactory
