"""
Corpus provider for the Wikipedia snapshot used by Attributed-QA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, Context, configure
from flexrag.common.misc import download_and_extract

from ..reader import LineDelimitedReader
from .corpus_dataset import CORPORA

_RESOURCES = "https://storage.googleapis.com/gresearch/attributed_language_models/wikipedia.zip"  # fmt: skip


@configure
class WikipediaAttributedQACorpusConfig:
    data_path: Optional[str] = None
    load_in_memory: bool = False


@CORPORA("wikipedia_attributedqa", config_class=WikipediaAttributedQACorpusConfig)
class WikipediaAttributedQACorpus:
    def __init__(self, config: WikipediaAttributedQACorpusConfig):
        if config.data_path is None:
            self._data_path = (
                FLEXRAG_CACHE_DIR / "corpora" / "enwiki_2021_attributedqa" / "wikipedia"
            )
        else:
            self._data_path = Path(config.data_path) / "wikipedia"
        if not self._data_path.exists():
            download_and_extract(_RESOURCES, self._data_path.parent.as_posix())
        self._data_files = list(self._data_path.glob("*.jsonl"))
        self._contexts: dict[str, Context] | None = None
        if config.load_in_memory:
            self._contexts = {}
            for context in self._iter_contexts():
                self._contexts[context.context_id] = context
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
            yield from self._contexts.values()
            return
        yield from self._iter_contexts()
        return

    @property
    def contexts(self) -> Mapping[str, Context]:
        if self._contexts is None:
            raise RuntimeError(
                "WikipediaAttributedQACorpus.contexts requires load_in_memory=True."
            )
        return self._contexts

    @property
    def context_ids(self) -> Iterator[str]:
        if self._contexts is not None:
            yield from self._contexts.keys()
            return
        for context in self._iter_contexts():
            yield context.context_id
        return
