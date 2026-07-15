"""
Corpus provider for the Wikipedia snapshot distributed by
`facebookresearch/DPR <https://github.com/facebookresearch/DPR>`_.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, Context, configure
from flexrag.common.misc import download

from ..reader import LineDelimitedReader
from .corpus_dataset import _InMemoryMappingCorpus

_RESOURCES = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"


@configure
class WikipediaDPRCorpusConfig:
    """Configuration for :class:`WikipediaDPRCorpus`.

    :param data_path: Local directory containing the DPR Wikipedia snapshot.
        If omitted, the corpus is stored under the FlexRAG cache directory.
    :type data_path: Optional[str]
    :param load_in_memory: Whether to build an in-memory mapping of all contexts.
        Defaults to False.
    :type load_in_memory: bool
    """

    data_path: Optional[str] = None
    load_in_memory: bool = False


class WikipediaDPRCorpus(_InMemoryMappingCorpus):
    """Wikipedia corpus backed by the DPR passage dataset.

    The corpus can always be iterated. Mapping-style access through ``contexts``
    and ``__len__`` requires ``load_in_memory=True``.
    """

    def __init__(self, config: WikipediaDPRCorpusConfig):
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "corpora" / "enwiki_2018_dpr"
        else:
            data_path = Path(config.data_path)
        self._text_file = data_path / "psgs_w100.tsv.gz"
        if not self._text_file.exists():
            download(_RESOURCES, self._text_file.parent.as_posix())
        if config.load_in_memory:
            contexts = {}
            for context in self._iter_contexts():
                contexts[context.context_id] = context
            self._set_materialized_contexts(contexts)
        return

    def _iter_contexts(self) -> Iterator[Context]:
        reader = LineDelimitedReader(
            self._text_file,
            titles=["id", "text", "title"],
            encoding="utf-8",
            file_format="tsv",
        )
        for data in reader:
            yield Context(
                context_id=data["id"],
                data={"title": data.get("title", ""), "text": data.get("text", "")},
            )
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
