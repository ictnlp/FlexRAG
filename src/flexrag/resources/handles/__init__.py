from __future__ import annotations

from .base import TypedHandle
from .chunker import ChunkerHandle
from .collection_backend import CollectionBackendHandle
from .context_store import ContextStoreHandle
from .encoder import EncoderHandle
from .generator import GeneratorHandle
from .ranker import RankerHandle
from .refiner import RefinerHandle
from .scorer import ScorerHandle
from .tokenizer import TokenizerHandle

HANDLE_TYPES: dict[str, type[TypedHandle]] = {
    "encoder": EncoderHandle,
    "generator": GeneratorHandle,
    "scorer": ScorerHandle,
    "ranker": RankerHandle,
    "refiner": RefinerHandle,
    "chunker": ChunkerHandle,
    "context_store": ContextStoreHandle,
    "collection_backend": CollectionBackendHandle,
    "tokenizer": TokenizerHandle,
}

__all__ = [
    "ChunkerHandle",
    "CollectionBackendHandle",
    "ContextStoreHandle",
    "EncoderHandle",
    "GeneratorHandle",
    "HANDLE_TYPES",
    "RankerHandle",
    "RefinerHandle",
    "ScorerHandle",
    "TokenizerHandle",
    "TypedHandle",
]
