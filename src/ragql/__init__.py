# SPDX-License-Identifier: MPL-2.0
# src/ragql/__init__.py
"""
ragql – Retrieval-Augmented Generation Query Language
"""

from __future__ import annotations
from importlib.metadata import version, PackageNotFoundError

# Public API:
from .core import RagQL  # main façade
from .loaders import REGISTRY  # dynamic list of loader callables

__all__ = [
    "RagQL",
    "REGISTRY",
    "__version__",
]

# Package metadata:
try:
    __version__: str = version("ragql")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
