"""
Helpers for loading Wikipedia corpus provided by
`google-research-datasets/Attributed-QA <https://github.com/google-research-datasets/Attributed-QA>`_.
"""

from os import PathLike
from pathlib import Path
from typing import Optional

from flexrag.common import FLEXRAG_CACHE_DIR
from flexrag.common.misc import download_and_extract

from .corpus_dataset import IterableCorpus, MappingCorpus

RESOURCES = (
    "https://storage.googleapis.com/gresearch/attributed_language_models/wikipedia.zip"
)


def load_wikipedia_attributedqa_corpus(
    data_path: Optional[PathLike] = None, load_in_memory: bool = False
) -> IterableCorpus | MappingCorpus:
    """
    Load the Wikipedia corpus provided by Attributed-QA.

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
        data_path = FLEXRAG_CACHE_DIR / "corpora" / "enwiki_2021_attributedqa" / "wikipedia"  # fmt: skip
    else:
        data_path = Path(data_path) / "wikipedia"
    if not data_path.exists():
        download_and_extract(RESOURCES, data_path.parent.as_posix())
    data_files = list(data_path.glob("*.jsonl"))

    # Load the corpus
    if load_in_memory:
        corpus = MappingCorpus.from_files(file_paths=data_files, id_field="id")
    else:
        corpus = IterableCorpus.from_files(file_paths=data_files, id_field="id")
    return corpus
