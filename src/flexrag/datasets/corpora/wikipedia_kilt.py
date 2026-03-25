"""
Corpus provider for the KILT knowledge source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, Context, configure
from flexrag.common.misc import download

from ..reader import LineDelimitedReader
from .corpus_dataset import CORPORA

_RESOURCES = "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"


@configure
class WikipediaKILTCorpusConfig:
    """Configuration for :class:`WikipediaKILTCorpus`.

    :param data_path: Local directory containing the KILT knowledge source file.
        If omitted, the corpus is stored under the FlexRAG cache directory.
    :type data_path: Optional[str]
    :param load_in_memory: Whether to build an in-memory mapping of all contexts.
        Defaults to False.
    :type load_in_memory: bool
    """

    data_path: Optional[str] = None
    load_in_memory: bool = False


@CORPORA("wikipedia_kilt", config_class=WikipediaKILTCorpusConfig)
class WikipediaKILTCorpus:
    """Wikipedia corpus backed by the KILT knowledge source.

    The corpus can always be iterated. Mapping-style access through ``contexts``
    and ``__len__`` requires ``load_in_memory=True``.
    """

    def __init__(self, config: WikipediaKILTCorpusConfig):
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "corpora" / "enwiki_2019_kilt"
        else:
            data_path = Path(config.data_path)
        self._text_file = data_path / "kilt_knowledgesource.json"
        if not self._text_file.exists():
            download(_RESOURCES, self._text_file.parent.as_posix())
        self._contexts: dict[str, Context] | None = None
        if config.load_in_memory:
            self._contexts = {}
            for context in self._iter_contexts():
                self._contexts[context.context_id] = context
        return

    def _iter_contexts(self) -> Iterator[Context]:
        reader = LineDelimitedReader(self._text_file)
        for data in reader:
            yield Context(
                context_id=data["_id"],
                data={
                    "text": data.get("text", ""),
                    "wikipedia_title": data.get("wikipedia_title", ""),
                },
            )
        return

    def __iter__(self) -> Iterator[Context]:
        if self._contexts is not None:
            yield from self._contexts.values()
            return
        yield from self._iter_contexts()
        return

    def __len__(self) -> int:
        if self._contexts is None:
            raise RuntimeError(
                "WikipediaKILTCorpus.__len__ requires load_in_memory=True."
            )
        return len(self._contexts)

    @property
    def contexts(self) -> Mapping[str, Context]:
        if self._contexts is None:
            raise RuntimeError(
                "WikipediaKILTCorpus.contexts requires load_in_memory=True."
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
