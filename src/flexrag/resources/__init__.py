"""Managed resources and runtime targets."""

from . import registrations as _registrations  # noqa: F401, E402
from .errors import RuntimeCallError
from .handles import (
    HANDLE_TYPES,
    ChunkerHandle,
    CollectionBackendHandle,
    ContextStoreHandle,
    EncoderHandle,
    GeneratorHandle,
    RankerHandle,
    RefinerHandle,
    ScorerHandle,
    TokenizerHandle,
    TypedHandle,
)
from .refs import ResourceRefDescriptor, ResourcesConfig, ResourceSpec, RuntimeConfig
from .registry import ResourceEntry, Resources, _ResourceRegister
from .resource_manager import ResourceManager

__all__ = [
    "ChunkerHandle",
    "CollectionBackendHandle",
    "ContextStoreHandle",
    "EncoderHandle",
    "GeneratorHandle",
    "HANDLE_TYPES",
    "RankerHandle",
    "RefinerHandle",
    "ResourceEntry",
    "ResourceManager",
    "ResourceRefDescriptor",
    "ResourceSpec",
    "Resources",
    "ResourcesConfig",
    "RuntimeCallError",
    "RuntimeConfig",
    "ScorerHandle",
    "TokenizerHandle",
    "TypedHandle",
    "_ResourceRegister",
]
