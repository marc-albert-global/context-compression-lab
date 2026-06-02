"""Importing this package registers every built-in compressor."""

from . import external, extractive, formats, lexical  # noqa: F401
from .base import Compressor, available, get, register

__all__ = ["Compressor", "available", "get", "register"]
