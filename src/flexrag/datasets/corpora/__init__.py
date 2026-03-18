"""Corpus protocols and concrete corpus providers."""

from .corpus_dataset import CORPORA, IterableCorpus, MappingCorpus
from .wikipedia_atlas import WikipediaAtlasCorpus, WikipediaAtlasCorpusConfig
from .wikipedia_attributeqa import (
    WikipediaAttributedQACorpus,
    WikipediaAttributedQACorpusConfig,
)
from .wikipedia_dpr import WikipediaDPRCorpus, WikipediaDPRCorpusConfig
from .wikipedia_kilt import WikipediaKILTCorpus, WikipediaKILTCorpusConfig
from .wikipedia_wikimedia import (
    WikipediaWikimediaCorpus,
    WikipediaWikimediaCorpusConfig,
)

__all__ = [
    "CORPORA",
    "IterableCorpus",
    "MappingCorpus",
    "WikipediaAtlasCorpus",
    "WikipediaAtlasCorpusConfig",
    "WikipediaAttributedQACorpus",
    "WikipediaAttributedQACorpusConfig",
    "WikipediaDPRCorpus",
    "WikipediaDPRCorpusConfig",
    "WikipediaKILTCorpus",
    "WikipediaKILTCorpusConfig",
    "WikipediaWikimediaCorpus",
    "WikipediaWikimediaCorpusConfig",
]
