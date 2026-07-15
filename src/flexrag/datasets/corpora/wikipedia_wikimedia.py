"""
Corpus provider for the Wikimedia Wikipedia dataset on Hugging Face.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Annotated, Optional

import orjson
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, Context, configure

from .corpus_dataset import _InMemoryMappingCorpus


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


class WikipediaWikimediaCorpus(_InMemoryMappingCorpus):
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
        contexts = {}
        for context in self._iter_from_dataset(raw_dataset):
            contexts[context.context_id] = context
        self._set_materialized_contexts(contexts)
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
                metadata={"url": item.get("url", "")},
            )
        return

    def __iter__(self) -> Iterator[Context]:
        assert self._ordered_contexts is not None
        yield from self._ordered_contexts
        return

    @property
    def context_ids(self) -> Iterator[str]:
        yield from self.contexts.keys()
        return


@configure
class WikipediaStructuredWikimediaCorpusConfig:
    """Configuration for :class:`WikipediaStructuredWikimediaCorpus`.

    :param data_path: Local directory containing the structured Wikimedia
        dataset snapshot. If omitted, the dataset is stored under the FlexRAG
        cache directory.
    :type data_path: Optional[str]
    :param subset: The structured Wikimedia subset to load. Available choices are
        ``20240916.en`` and ``20240916.fr``.
    :type subset: str
    :param context_mode: How contexts are organized. Available choices are
        ``section`` and ``document``.
    :type context_mode: str
    """

    data_path: Optional[str] = None
    subset: Annotated[str, Choices("20240916.en", "20240916.fr")] = "20240916.en"
    context_mode: Annotated[str, Choices("section", "document")] = "section"


class WikipediaStructuredWikimediaCorpus:
    """Wikipedia corpus backed by Wikimedia Structured Wikipedia on Hugging Face."""

    def __init__(self, config: WikipediaStructuredWikimediaCorpusConfig):
        self._config = config
        if config.data_path is None:
            self._repo_dir = FLEXRAG_CACHE_DIR / "corpora" / "structured-wikipedia"
        else:
            self._repo_dir = Path(config.data_path)
        if not self._repo_dir.exists():
            snapshot_download(
                repo_id="wikimedia/structured-wikipedia",
                repo_type="dataset",
                local_dir=self._repo_dir.as_posix(),
            )
        self._subset_dir = self._repo_dir / config.subset
        if not self._subset_dir.exists():
            raise FileNotFoundError(
                f"Structured Wikipedia subset not found: {self._subset_dir}"
            )
        return

    def _iter_items(self) -> Iterator[dict]:
        for zip_path in sorted(self._subset_dir.glob("*.zip")):
            with zipfile.ZipFile(zip_path) as archive:
                for member in sorted(archive.namelist()):
                    if not member.endswith(".jsonl"):
                        continue
                    if member.startswith("__MACOSX/"):
                        continue
                    if PurePosixPath(member).name.startswith("._"):
                        continue
                    with archive.open(member, "r") as f:
                        for line in f:
                            yield orjson.loads(line)
        return

    def _iter_values(self, part: dict) -> Iterator[str]:
        value = part.get("value", "")
        if value:
            yield value
        for child in part.get("has_parts", []):
            yield from self._iter_values(child)
        return

    def __iter__(self) -> Iterator[Context]:
        for item in self._iter_items():
            if self._config.context_mode == "document":
                texts = []
                for section in item["sections"]:
                    text = "\n".join(self._iter_values(section))
                    if text:
                        texts.append(text)
                if texts:
                    yield Context(
                        context_id=str(item["identifier"]),
                        data={"title": item["name"], "text": "\n\n".join(texts)},
                        source="wikimedia/structured-wikipedia",
                        metadata={"url": item["url"]},
                    )
                continue
            for section_idx, section in enumerate(item["sections"]):
                text = "\n".join(self._iter_values(section))
                if not text:
                    continue
                yield Context(
                    context_id=f"{item['identifier']}:{section_idx}",
                    data={
                        "title": item["name"],
                        "section": section["name"],
                        "text": text,
                    },
                    source="wikimedia/structured-wikipedia",
                    metadata={"url": item["url"]},
                )
        return

    @property
    def context_ids(self) -> Iterator[str]:
        for context in self:
            assert context.context_id is not None
            yield context.context_id
        return
