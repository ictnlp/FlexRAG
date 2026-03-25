"""
Corpus provider for the Wikimedia Wikipedia dataset on Hugging Face.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Context, configure

from .corpus_dataset import CORPORA


@configure
class WikipediaWikimediaCorpusConfig:
    """Configuration for :class:`WikipediaWikimediaCorpus`.

    :param data_path: Local directory containing the Wikimedia dataset snapshot.
        If omitted, the dataset is stored under the FlexRAG cache directory.
    :type data_path: Optional[str]
    :param subset: The Wikimedia subset to load, e.g. ``20231101.en``.
    :type subset: str
    """

    data_path: Optional[str] = None
    subset: str = "20231101.en"


@CORPORA("wikipedia_wikimedia", config_class=WikipediaWikimediaCorpusConfig)
class WikipediaWikimediaCorpus:
    """Wikipedia corpus backed by the Wikimedia dataset on Hugging Face.

    This corpus always materializes its contexts in memory, so mapping access
    and ``__len__`` are always available.
    """

    def __init__(self, config: WikipediaWikimediaCorpusConfig):
        self._config = config
        if config.data_path is None:
            self._repo_dir = FLEXRAG_CACHE_DIR / "corpora" / "wikimedia"
        else:
            self._repo_dir = Path(config.data_path)
        if not self._repo_dir.exists():
            snapshot_download(
                repo_id="wikimedia/wikipedia",
                repo_type="dataset",
                local_dir=self._repo_dir.as_posix(),
            )
        raw_dataset = load_dataset(
            path=self._repo_dir.as_posix(),
            name=self._config.subset,
            split="train",
        )
        self._contexts: dict[str, Context] = {}
        for context in self._iter_from_dataset(raw_dataset):
            self._contexts[context.context_id] = context
        return

    def _iter_from_dataset(self, dataset: Dataset) -> Iterator[Context]:
        for item in dataset:
            yield Context(
                context_id=str(item["id"]),
                data={
                    "title": item.get("title", ""),
                    "text": item.get("text", ""),
                },
                source="wikimedia/wikipedia",
                meta_data={"url": item.get("url", "")},
            )
        return

    def __iter__(self) -> Iterator[Context]:
        yield from self._contexts.values()
        return

    def __len__(self) -> int:
        return len(self._contexts)

    @property
    def contexts(self) -> Mapping[str, Context]:
        return self._contexts

    @property
    def context_ids(self) -> Iterator[str]:
        yield from self._contexts.keys()
        return
