"""
The corpora submodule provides helper classes for loading public corpora.
"""

from .corpus_dataset import IterableCorpus, MappingCorpus
from .wikipedia_atlas import load_wikipedia_atlas_corpus
from .wikipedia_kilt import load_wikipedia_kilt_corpus

__all__ = [
    "IterableCorpus",
    "MappingCorpus",
    "load_wikipedia_atlas_corpus",
    "load_wikipedia_kilt_corpus",
]
