"""
Corpus provider for Wikipedia snapshots distributed by
`facebookresearch/atlas <https://github.com/facebookresearch/atlas>`_.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Iterator, Mapping, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, Context, configure
from flexrag.common.misc import download

from ..reader import LineDelimitedReader
from .corpus_dataset import CORPORA

_RESOURCES = {
    "enwiki_2017_atlas": {
        "infobox": "https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-dec2017/infobox.jsonl",
        "text": "https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-dec2017/text-list-100-sec.jsonl",
    },
    "enwiki_2018_atlas": {
        "infobox": "https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-dec2018/infobox.jsonl",
        "text": "https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl",
    },
    "enwiki_2019_atlas": {
        "infobox": "https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-aug2019/infobox.jsonl",
        "text": "https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-aug2019/text-list-100-sec.jsonl",
    },
    "enwiki_2020_atlas": {
        "infobox": "https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-dec2020/infobox.jsonl",
        "text": "https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-dec2020/text-list-100-sec.jsonl",
    },
    "enwiki_2021_atlas": {
        "infobox": "https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-dec2021/infobox.jsonl",
        "text": "https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl",
    },
}


@configure
class WikipediaAtlasCorpusConfig:
    data_path: Optional[str] = None
    data_version: Annotated[
        str,
        Choices(
            "enwiki_2017_atlas",
            "enwiki_2018_atlas",
            "enwiki_2019_atlas",
            "enwiki_2020_atlas",
            "enwiki_2021_atlas",
        ),
    ] = "enwiki_2021_atlas"
    load_in_memory: bool = False
    include_infobox: bool = True


@CORPORA("wikipedia_atlas", config_class=WikipediaAtlasCorpusConfig)
class WikipediaAtlasCorpus:
    def __init__(self, config: WikipediaAtlasCorpusConfig):
        self._config = config
        self._file_paths = self._ensure_files(config)
        self._contexts: dict[str, Context] | None = None
        if config.load_in_memory:
            self._contexts = {}
            for context in self._iter_contexts():
                self._contexts[context.context_id] = context
        return

    @staticmethod
    def _ensure_files(config: WikipediaAtlasCorpusConfig) -> list[Path]:
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "corpora" / config.data_version
        else:
            data_path = Path(config.data_path)
        text_file = data_path / "text-list-100-sec.jsonl"
        infobox_file = data_path / "infobox.jsonl"
        file_paths = []
        if not text_file.exists():
            download(
                _RESOURCES[config.data_version]["text"], text_file.parent.as_posix()
            )
        file_paths.append(text_file)
        if config.include_infobox:
            if not infobox_file.exists():
                download(
                    _RESOURCES[config.data_version]["infobox"],
                    infobox_file.parent.as_posix(),
                )
            file_paths.append(infobox_file)
        return file_paths

    def _iter_contexts(self) -> Iterator[Context]:
        for file_path in self._file_paths:
            reader = LineDelimitedReader(file_path=file_path)
            for data in reader:
                context_id = data["id"]
                yield Context(
                    context_id=context_id,
                    data={
                        "title": data.get("title", ""),
                        "section": data.get("section", ""),
                        "text": data.get("text", ""),
                    },
                )
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
                "WikipediaAtlasCorpus.contexts requires load_in_memory=True."
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
