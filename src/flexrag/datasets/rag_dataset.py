from dataclasses import dataclass, field
from typing import Iterator, Optional

from flexrag.common_dataclass import Context, RAGEvalData
from flexrag.text_process import TextProcessPipeline, TextProcessPipelineConfig
from flexrag.utils import LOGGER_MANAGER

from .hf_dataset import HFDataset, HFDatasetConfig
from .line_delimited_dataset import LineDelimitedDataset, LineDelimitedDatasetConfig

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.rag_dataset")


@dataclass
class RAGEvalDatasetConfig(HFDatasetConfig):
    """The configuration for ``RAGEvalDataset``.
    This dataset helps to load the evaluation dataset collected by `FlashRAG <https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets>`_.
    The ``__iter__`` method will yield `RAGEvalData` objects.

    For example, you can load the `test` set of the `NaturalQuestions` dataset by running the following code:

    .. code-block:: python

        from flexrag.datasets import RAGEvalDataset, RAGEvalDatasetConfig

        cfg = RAGEvalDatasetConfig(
            name="nq",
            split="test",
        )
        dataset = RAGEvalDataset(cfg)

    You can also load the dataset from a local repository by specifying the path.
    For example, you can download the dataset by running the following command:

        >>> git lfs install
        >>> git clone https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets flashrag

    Then you can load the dataset by running the following code:

    .. code-block:: python

        from flexrag.datasets import RAGEvalDataset, RAGEvalDatasetConfig

        cfg = RAGEvalDatasetConfig(
            path="json",
            data_files=["flashrag/nq/test.jsonl"],
            split="train",
        )
        dataset = RAGEvalDataset(cfg)

    Available datasets include:

        - 2wikimultihopqa: dev, train
        - ambig_qa: dev, train
        - arc: dev, test, train
        - asqa: dev, train
        - ay2: dev, train
        - bamboogle: test
        - boolq: dev, train
        - commonsenseqa: dev, train
        - curatedtrec: test, train
        - eli5: dev, train
        - fermi: dev, test, train
        - fever: dev, train
        - hellaswag: dev, train
        - hotpotqa: dev, train
        - mmlu: 5_shot, dev, test, train
        - msmarco-qa: dev, train
        - musique: dev, train
        - narrativeqa: dev, test, train
        - nq: dev, test, train
        - openbookqa: dev, test, train
        - piqa: dev, train
        - popqa: test
        - quartz: dev, test, train
        - siqa: dev, train
        - squad: dev, train
        - t-rex: dev, train
        - triviaqa: dev, test, train
        - truthful_qa: dev
        - web_questions: test, train
        - wikiasp: dev, test, train
        - wikiqa: dev, test, train
        - wned: dev
        - wow: dev, train
        - zero-shot_re: dev, train
    """

    path: str = "RUC-NLPIR/FlashRAG_datasets"


class RAGEvalDataset(HFDataset):
    """The dataset for loading RAG evaluation data."""

    def __init__(self, cfg: RAGEvalDatasetConfig) -> None:
        super().__init__(cfg)
        return

    def __getitem__(self, index: int) -> RAGEvalData:
        data = super().__getitem__(index)
        golden_contexts = data.pop("golden_contexts", None)
        golden_contexts = (
            [Context(**context) for context in golden_contexts]
            if golden_contexts is not None
            else None
        )
        formatted_data = RAGEvalData(
            question=data.pop("question"),
            golden_contexts=golden_contexts,
            golden_answers=data.pop("golden_answers", None),
        )
        formatted_data.meta_data = data.pop("meta_data", {})
        formatted_data.meta_data.update(data)
        return formatted_data

    def __iter__(self) -> Iterator[RAGEvalData]:
        yield from super().__iter__()


@dataclass
class RAGCorpusDatasetConfig(LineDelimitedDatasetConfig):
    """The configuration for ``RAGCorpusDataset``.
    This dataset helps to load the pre-processed corpus data for RAG retrieval.
    The ``__iter__`` method will yield `Context` objects.

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

        from flexrag.datasets import RAGCorpusDataset, RAGCorpusDatasetConfig
        from flexrag.text_process import TextProcessPipelineConfig, LengthFilterConfig

        cfg = RAGCorpusDatasetConfig(
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
        dataset = RAGCorpusDataset(cfg)

    The above code will load the corpus data from the provided files and preprocess the `text` field with a length filter.
    For any text with a length less than 10 or greater than 4096 characters, it will be filtered out.
    """

    saving_fields: list[str] = field(default_factory=list)
    id_field: Optional[str] = None
    processors: dict[str, TextProcessPipelineConfig] = field(default_factory=dict)  # type: ignore


class RAGCorpusDataset(LineDelimitedDataset):
    """The dataset for loading pre-processed corpus data for RAG retrieval."""

    def __init__(self, cfg: RAGCorpusDatasetConfig) -> None:
        super().__init__(cfg)
        # load arguments
        self.saving_fields = cfg.saving_fields
        self.id_field = cfg.id_field
        if self.id_field is None:
            logger.warning("No id field is provided, using the index as the id field")

        # load processors for each fields
        assert all(
            key in self.saving_fields for key in cfg.processors
        ), f"The field to process is not in the saving fields: {self.saving_fields}."
        self.processors = {
            key: TextProcessPipeline(cfg.processors[key]) for key in cfg.processors
        }
        return

    def __iter__(self) -> Iterator[Context]:
        for n, data in enumerate(super().__iter__()):
            # prepare context_id
            if self.id_field is not None:
                context_id = data.pop(self.id_field)
            else:
                context_id = str(n)

            # remove unused fields
            if len(self.saving_fields) > 0:
                data = {key: data.get(key, "") for key in self.saving_fields}

            # preprocess each fields
            for key in data:
                if key in self.processors:
                    data[key] = self.processors[key](data[key])

            # filter the data
            if any(data[key] is None for key in data):
                continue

            yield Context(context_id=context_id, data=data)
