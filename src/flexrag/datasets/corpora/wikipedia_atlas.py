"""
Helpers for loading Wikipedia corpus provided by
`facebookresearch/atlas <https://github.com/facebookresearch/atlas>`_.
"""

from os import PathLike
from pathlib import Path
from typing import Literal, Optional

from flexrag.common import FLEXRAG_CACHE_DIR
from flexrag.common.misc import download

from .corpus_dataset import IterableCorpus, MappingCorpus

RESOURCES = {
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


def load_wikipedia_atlas_corpus(
    data_path: Optional[PathLike] = None,
    data_version: Literal[
        "enwiki_2017_atlas",
        "enwiki_2018_atlas",
        "enwiki_2019_atlas",
        "enwiki_2020_atlas",
        "enwiki_2021_atlas",
    ] = "enwiki_2021_atlas",
    load_in_memory: bool = False,
    include_infobox: bool = True,
) -> IterableCorpus | MappingCorpus:
    """
    Load the Wikipedia corpus provided by Atlas.

    :param data_path: The path to the data directory.
        If None, the data will be downloaded to the default cache directory.
    :type data_path: Optional[PathLike]
    :param data_version: The version of the Wikipedia corpus to load.
        Defaults to "enwiki_2021_atlas".
        Available choices are: `enwiki_2017_atlas`, `enwiki_2018_atlas`,
        `enwiki_2019_atlas`, `enwiki_2020_atlas`, `enwiki_2021_atlas`.
    :type data_version: str
    :param load_in_memory: Whether to load the corpus into memory. Defaults to False.
    :type load_in_memory: bool
    :param include_infobox: Whether to include the infobox data. Defaults to True.
    :type include_infobox: bool
    :return: The loaded corpus.
    :rtype: IterableCorpus | MappingCorpus
    """
    # Download the corpus if not exists
    if data_path is None:
        data_path = FLEXRAG_CACHE_DIR / "corpora" / data_version
    else:
        data_path = Path(data_path)
    text_file = data_path / "text-list-100-sec.jsonl"
    infobox_file = data_path / "infobox.jsonl"
    file_paths = []
    if not text_file.exists():
        download(RESOURCES[data_version]["text"], text_file.parent.as_posix())
    file_paths.append(text_file)
    if include_infobox:
        if not infobox_file.exists():
            download(RESOURCES[data_version]["infobox"], infobox_file.parent.as_posix())
        file_paths.append(infobox_file)

    # Load the corpus
    if load_in_memory:
        corpus = MappingCorpus.from_files(
            file_paths=file_paths,
            saving_fields=["title", "section", "text"],
            id_field="id",
        )
    else:
        corpus = IterableCorpus.from_files(
            file_paths=file_paths,
            saving_fields=["title", "section", "text"],
            id_field="id",
        )
    return corpus
