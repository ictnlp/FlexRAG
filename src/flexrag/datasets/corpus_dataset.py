from __future__ import annotations

from dataclasses import field
from glob import glob
from pathlib import Path
from typing import Iterator, Optional

from flexrag.text_process import TextProcessPipeline, TextProcessPipelineConfig
from flexrag.utils import LOGGER_MANAGER, configure
from flexrag.utils.dataclasses import Context

from .dataset import IterableDataset, MappingDataset
from .reader import LineDelimitedReader

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.rag_dataset")


@configure
class IterableCorpusConfig:
    """The configuration for ``IterableCorpus``.
    This dataset helps to load the pre-processed corpus data for retrieval.
    The ``__iter__`` method will yield `Context` objects.

    :param file_paths: The paths to the line delimited files.
        It supports unix style path pattern.
    :type file_paths: list[str]
    :param titles: The field names of the corpus data.
        This option is only used when the corpus is in tsv or csv format.
        If not specified, the field names will be inferred from the first line of the file.
    :type titles: list[str]
    :param encoding: The encoding of the files.
    :type encoding: str
    :param saving_fields: The fields to save in the context. If not specified, all fields will be saved.
    :type saving_fields: list[str]
    :param id_field: The field to use as the context_id. If not specified, the ordinal number will be used.
    :type id_field: Optional[str]
    :param processors: The preprocessors for each field. Default is {}.
        The key is the field name, and the value is the `TextProcessPipelineConfig`.
    :type processors: dict[str, TextProcessPipelineConfig]

    For example, to load the corpus provided by the `Atlas <https://github.com/facebookresearch/atlas>`_,
    you can download the corpus by running the following command:

    .. code-block:: bash

        wget https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl
        wget https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-dec2021/infobox.jsonl

    Then you can use the following code to load the corpus with a length filter:

    .. code-block:: python

        from flexrag.datasets import IterableCorpus, IterableCorpusConfig
        from flexrag.text_process import TextProcessPipelineConfig, LengthFilterConfig

        cfg = IterableCorpusConfig(
            file_paths=[
                "/data/zhangzhuocheng/Lab/Python/LLM/datasets/RAG/wikipedia/wiki_2021/infobox.jsonl",
                "/data/zhangzhuocheng/Lab/Python/LLM/datasets/RAG/wikipedia/wiki_2021/text-list-100-sec.jsonl",
            ],
            saving_fields=["title", "text"],
            processors={
                "text": TextProcessPipelineConfig(
                    processor_type=["length_filter"],
                    length_filter_config=LengthFilterConfig(
                        max_chars=4096,
                        min_chars=10,
                    ),
                )
            },
            encoding="utf-8",
        )
        dataset = IterableCorpus(cfg)

    The above code will load the corpus data from the provided files and preprocess the `text` field with a length filter.
    For any text with a length less than 10 or greater than 4096 characters, it will be filtered out.
    """

    file_paths: list[str]
    titles: list[str] = field(default_factory=list)
    encoding: str = "utf-8"
    saving_fields: list[str] = field(default_factory=list)
    id_field: Optional[str] = None
    processors: dict[str, TextProcessPipelineConfig] = field(default_factory=dict)  # type: ignore


class IterableCorpus(IterableDataset):
    """The helper dataset for loading pre-processed corpus data for retrieval."""

    def __init__(self, cfg: IterableCorpusConfig) -> None:
        super().__init__(cfg)
        # set up data paths
        self.file_paths: list[Path] = []
        for p in cfg.file_paths:
            if isinstance(p, str):
                paths = [Path(p_) for p_ in glob(p)]
            else:
                raise TypeError(f"Unsupported file path type: {type(p)}")
            if len(paths) == 0:
                raise FileNotFoundError(f"File {p} does not exist.")
            self.file_paths.extend(paths)

        # set up reader parameters
        self.encoding = cfg.encoding
        self.titles = cfg.titles

        # set up other parameters
        self.saving_fields = cfg.saving_fields
        self.id_field = cfg.id_field
        if self.id_field is None:
            logger.warning("No id field is provided, using the index as the id field")

        # load processors for each fields
        if len(self.saving_fields) > 0:
            assert all(
                key in self.saving_fields for key in cfg.processors
            ), f"The field to process is not in the saving fields: {self.saving_fields}."
        self.processors = {
            key: TextProcessPipeline(cfg.processors[key]) for key in cfg.processors
        }
        return

    def __iter__(self) -> Iterator[Context]:
        n = 0
        for file_path in self.file_paths:
            reader = LineDelimitedReader(
                file_path=file_path,
                titles=self.titles,
                encoding=self.encoding,
            )
            for data in reader:
                # prepare context_id
                if self.id_field is not None:
                    context_id = data.pop(self.id_field)
                else:
                    context_id = str(n)
                n += 1

                # remove unused fields
                if len(self.saving_fields) > 0:
                    data = {key: data.get(key, "") for key in self.saving_fields}

                # preprocess each fields
                for key, processor in self.processors.items():
                    if key in data:
                        data[key] = self.processors[key](data[key])

                # filter the data
                if any(data[key] is None for key in data):
                    continue

                yield Context(context_id=context_id, data=data)

    def to_mapping_corpus(self, mmap: bool = True) -> MappingCorpus:
        """Convert the iterable corpus to a mapping corpus.

        :param mmap: Whether to use memory-mapped file for the mapping corpus.
            If set to True, it will create a memory-mapped file for the corpus.
            Otherwise, the corpus will be loaded into memory.
        :type mmap: bool
        :return: The mapping corpus.
        :rtype: MappingCorpus
        """
        raise NotImplementedError


class MappingCorpus(MappingDataset): ...
