"""
Helpers for loading Wikipedia corpus provided by
`facebookresearch/DPR <https://github.com/facebookresearch/DPR>`_.
"""

from os import PathLike
from pathlib import Path
from typing import Optional

from flexrag.common import FLEXRAG_CACHE_DIR
from flexrag.common.misc import download

from .corpus_dataset import IterableCorpus, MappingCorpus

RESOURCES = {
    "dpr": "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
}


def load_wikipedia_dpr_corpus(
    data_path: Optional[PathLike] = None, load_in_memory: bool = False
) -> IterableCorpus | MappingCorpus:
    """
    Load the Wikipedia corpus provided by DPR.

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
        data_path = FLEXRAG_CACHE_DIR / "corpora" / "enwiki_2018_dpr"
    else:
        data_path = Path(data_path)
    text_file = data_path / "psgs_w100.tsv.gz"
    if not text_file.exists():
        download(RESOURCES["dpr"], text_file.parent.as_posix())

    # Load the corpus
    if load_in_memory:
        corpus = MappingCorpus.from_files(file_paths=[text_file], id_field="id")
    else:
        corpus = IterableCorpus.from_files(file_paths=[text_file], id_field="id")
    return corpus
