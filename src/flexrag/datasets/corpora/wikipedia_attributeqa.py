"""
Corpus provider for the Wikipedia snapshot used by Attributed-QA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, Context, configure
from flexrag.common.misc import download_and_extract

from ..reader import LineDelimitedReader
from .corpus_dataset import CORPORA, _InMemoryMappingCorpus

_RESOURCES = "https://storage.googleapis.com/gresearch/attributed_language_models/wikipedia.zip"  # fmt: skip


@configure
class WikipediaAttributedQACorpusConfig:
    """Configuration for :class:`WikipediaAttributedQACorpus`.

    :param data_path: Local directory containing the Attributed-QA Wikipedia
        dump. If omitted, the corpus is stored under the FlexRAG cache
        directory.
    :type data_path: Optional[str]
    :param load_in_memory: Whether to build an in-memory mapping of all contexts.
        Defaults to False.
    :type load_in_memory: bool
    """

    data_path: Optional[str] = None
    load_in_memory: bool = False


@CORPORA("wikipedia_attributedqa", config_class=WikipediaAttributedQACorpusConfig)
class WikipediaAttributedQACorpus(_InMemoryMappingCorpus):
    """Wikipedia corpus backed by the Attributed-QA Wikipedia snapshot.

    The corpus can always be iterated. Mapping-style access through ``contexts``
    and ``__len__`` requires ``load_in_memory=True``.
    """

    def __init__(self, config: WikipediaAttributedQACorpusConfig):
        if config.data_path is None:
            self._data_path = (
                FLEXRAG_CACHE_DIR / "corpora" / "enwiki_2021_attributedqa" / "wikipedia"
            )
        else:
            self._data_path = Path(config.data_path) / "wikipedia"
        if not self._data_path.exists():
            download_and_extract(_RESOURCES, self._data_path.parent.as_posix())
        self._data_files = sorted(self._data_path.glob("*.jsonl"))
        if config.load_in_memory:
            contexts = {}
            for context in self._iter_contexts():
                contexts[context.context_id] = context
            self._set_materialized_contexts(contexts)
        return

    def _iter_contexts(self) -> Iterator[Context]:
        for file_path in self._data_files:
            reader = LineDelimitedReader(file_path=file_path)
            for data in reader:
                payload = dict(data)
                context_id = payload.pop("id")
                yield Context(context_id=context_id, data=payload)
        return

    def __iter__(self) -> Iterator[Context]:
        if self._contexts is not None:
            assert self._ordered_contexts is not None
            yield from self._ordered_contexts
        else:
            yield from self._iter_contexts()
        return

    @property
    def context_ids(self) -> Iterator[str]:
        if self._contexts is not None:
            yield from self.contexts.keys()
        else:
            for context in self._iter_contexts():
                yield context.context_id
        return
