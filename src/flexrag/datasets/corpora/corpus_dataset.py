from __future__ import annotations

from glob import glob
from os import PathLike
from pathlib import Path
from typing import Iterator, Optional

from flexrag.common import LOGGER_MANAGER
from flexrag.common.dataclasses import Context
from flexrag.processors.text_processors import TextProcessPipeline

from ..core import IterableDataset, MappingDataset
from ..reader import LineDelimitedReader

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.rag_dataset")


class IterableCorpus(IterableDataset[Context]):
    """The helper dataset for loading pre-processed corpus data for retrieval."""

    def __init__(
        self,
        file_paths: list[PathLike],
        titles: list[str] = [],
        encoding: str = "utf-8",
        saving_fields: list[str] = [],
        meta_fields: list[str] = [],
        id_field: Optional[str] = None,
        processors: dict[str, TextProcessPipeline] = {},
    ) -> None:
        super().__init__()
        # set up data paths
        self.file_paths: list[Path] = []
        for p in file_paths:
            if isinstance(p, str):
                paths = [Path(p_) for p_ in glob(p)]
            elif isinstance(p, Path):
                paths = [p]
            else:
                raise TypeError(f"Unsupported file path type: {type(p)}")
            if len(paths) == 0:
                raise FileNotFoundError(f"File {p} does not exist.")
            self.file_paths.extend(paths)

        # set up reader parameters
        self.encoding = encoding
        self.titles = titles

        # set up other parameters
        self.saving_fields = saving_fields
        self.meta_fields = meta_fields
        self.id_field = id_field
        if self.id_field is None:
            logger.warning("No id field is provided, using the index as the id field")

        # load processors for each fields
        if len(self.saving_fields) > 0:
            assert all(
                key in self.saving_fields for key in processors
            ), f"The field to process is not in the saving fields: {self.saving_fields}."
        self.processors = processors
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

                # select saving fields
                if len(self.saving_fields) > 0:
                    data = {key: data.get(key, "") for key in self.saving_fields}
                # select meta fields
                if len(self.meta_fields) > 0:
                    meta_data = {key: data.pop(key, "") for key in self.meta_fields}
                else:
                    meta_data = {}

                # preprocess each fields
                for key, processor in self.processors.items():
                    if key in data:
                        data[key] = processor(data[key])

                # filter the data
                if any(data[key] is None for key in data):
                    continue

                yield Context(context_id=context_id, data=data, meta_data=meta_data)

    @classmethod
    def from_files(
        cls,
        file_paths: list[PathLike],
        titles: list[str] = [],
        encoding: str = "utf-8",
        saving_fields: list[str] = [],
        meta_fields: list[str] = [],
        id_field: Optional[str] = None,
        processors: dict[str, TextProcessPipeline] = {},
    ) -> IterableCorpus:
        """Create an IterableCorpus from the given file paths and parameters.

        :param file_paths: The paths to the line delimited files.
            It supports glob style path pattern.
        :type file_paths: list[PathLike]
        :param titles: The field names of the corpus data.
            This option is only used when the corpus is in `tsv` or `csv` format.
            If not specified, the field names will be inferred from the first line of the file.
        :type titles: list[str]
        :param encoding: The encoding of the files.
        :type encoding: str
        :param saving_fields: The fields to save in the context.
            If not specified, all fields will be saved in the `data` attribute.
        :type saving_fields: list[str]
        :param meta_fields: The fields to save in the meta_data of the context.
            If not specified, no fields will be saved in the `meta_data` attribute.
        :type meta_fields: list[str]
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

            from flexrag.datasets import IterableCorpus
            from flexrag.text_process import TextProcessPipelineConfig, LengthFilterConfig

            processors = {
                "text": TextProcessPipelineConfig(
                    processor_type=["length_filter"],
                    length_filter_config=LengthFilterConfig(
                        max_chars=4096,
                        min_chars=10,
                    ),
                )
            }
            dataset = IterableCorpus.from_files(
                file_paths=[
                    "/data/zhangzhuocheng/Lab/Python/LLM/datasets/RAG/wikipedia/wiki_2021/infobox.jsonl",
                    "/data/zhangzhuocheng/Lab/Python/LLM/datasets/RAG/wikipedia/wiki_2021/text-list-100-sec.jsonl",
                ],
                saving_fields=["title", "text"],
                processors=processors,
                encoding="utf-8",
            )

        The above code will load the corpus data from the provided files and preprocess the `text` field with a length filter.
        For any text with a length less than 10 or greater than 4096 characters, it will be filtered out.
        """
        return cls(
            file_paths=file_paths,
            titles=titles,
            encoding=encoding,
            saving_fields=saving_fields,
            id_field=id_field,
            processors=processors,
        )


class MappingCorpus(MappingDataset[Context]):
    """The helper dataset for loading pre-processed corpus data for retrieval."""

    def __init__(self, contexts: dict[str, Context]) -> None:
        super().__init__()
        self._contexts = contexts
        self._keys = sorted(contexts.keys())
        return

    @classmethod
    def from_iterator(cls, iterator: Iterator[Context]) -> MappingCorpus:
        """Create a MappingCorpus from the given context iterator.

        :param iterator: The iterator of Context objects.
        :type iterator: Iterator[Context]
        """
        contexts = {}
        for context in iterator:
            contexts[context.context_id] = context
        return cls(contexts)

    @classmethod
    def from_files(
        cls,
        file_paths: list[PathLike],
        titles: list[str] = [],
        encoding: str = "utf-8",
        saving_fields: list[str] = [],
        id_field: Optional[str] = None,
        processors: dict[str, TextProcessPipeline] = {},
    ) -> MappingCorpus:
        """Create an MappingCorpus from the given file paths and parameters.

        :param file_paths: The paths to the line delimited files.
            It supports glob style path pattern.
        :type file_paths: list[PathLike]
        :param titles: The field names of the corpus data.
            This option is only used when the corpus is in `tsv` or `csv` format.
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

            from flexrag.datasets import MappingCorpus
            from flexrag.text_process import TextProcessPipelineConfig, LengthFilterConfig

            processors = {
                "text": TextProcessPipelineConfig(
                    processor_type=["length_filter"],
                    length_filter_config=LengthFilterConfig(
                        max_chars=4096,
                        min_chars=10,
                    ),
                )
            }
            dataset = MappingCorpus.from_files(
                file_paths=[
                    "/data/zhangzhuocheng/Lab/Python/LLM/datasets/RAG/wikipedia/wiki_2021/infobox.jsonl",
                    "/data/zhangzhuocheng/Lab/Python/LLM/datasets/RAG/wikipedia/wiki_2021/text-list-100-sec.jsonl",
                ],
                saving_fields=["title", "text"],
                processors=processors,
                encoding="utf-8",
            )

        The above code will load the corpus data from the provided files and preprocess the `text` field with a length filter.
        For any text with a length less than 10 or greater than 4096 characters, it will be filtered out.
        """
        return cls.from_iterator(
            IterableCorpus.from_files(
                file_paths=file_paths,
                titles=titles,
                encoding=encoding,
                saving_fields=saving_fields,
                id_field=id_field,
                processors=processors,
            )
        )

    def get_item(self, index: int) -> Context:
        context_id = self._keys[index]
        return self._contexts[context_id]

    def get_by_id(self, context_id: str) -> Context:
        return self._contexts[context_id]

    def __len__(self) -> int:
        return len(self._contexts)
