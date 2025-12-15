"""
Helpers for loading Wikipedia corpus provided by
`facebookresearch/KILT <https://github.com/facebookresearch/KILT>`_.
"""

from os import PathLike
from pathlib import Path
from typing import Optional

from flexrag.common import FLEXRAG_CACHE_DIR
from flexrag.common.misc import download

from .corpus_dataset import IterableCorpus, MappingCorpus

RESOURCES = {"kilt": "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"}


def load_wikipedia_kilt_corpus(
    data_path: Optional[PathLike] = None, load_in_memory: bool = False
) -> IterableCorpus | MappingCorpus:
    """
    Load the Wikipedia corpus provided by KILT.

    :param data_path: The path to the data directory.
        If None, the data will be downloaded to the default cache directory.
    :type data_path: Optional[PathLike]
    :param load_in_memory: Whether to load the corpus into memory. Defaults to False.
    :type load_in_memory: bool
    :return: The loaded corpus.
    :rtype: IterableCorpus | MappingCorpus
    """
    # Download the corpus if not exists
    if data_path is None:
        data_path = FLEXRAG_CACHE_DIR / "corpora" / "enwiki_2019_kilt"
    else:
        data_path = Path(data_path)
    text_file = data_path / "kilt_knowledgesource.json"
    if not text_file.exists():
        download(RESOURCES["kilt"], text_file.parent.as_posix())

    # Load the corpus
    if load_in_memory:
        corpus = MappingCorpus.from_files(file_paths=[text_file], id_field="_id")
    else:
        corpus = IterableCorpus.from_files(file_paths=[text_file], id_field="_id")
    return corpus
