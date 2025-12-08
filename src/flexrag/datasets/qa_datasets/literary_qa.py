from pathlib import Path
from typing import Annotated

from flexrag.common import Choices, configure
from flexrag.common.dataclasses import Context

from ..reader import LineDelimitedReader
from .qa_dataset_base import KNOWLEDGE_QA_DATASETS, QA_DATASETS, KnowledgeQADatasetBase


@configure
class LiteraryQADatasetConfig:
    """Configuration for LiteraryQADataset.

    `LiteraryQA <https://arxiv.org/abs/2510.13494>`_ is a high-quality subset
    of NarrativeQA focused on literary works, designed to address issues with
    noisy documents and flawed QA pairs in the original benchmark.

    :param data_path: The path to the LiteraryQA dataset file. Required.
        The dataset could be obtained by following the instructions provided
        in the `official repository <https://github.com/SapienzaNLP/LiteraryQA>`_.
    :type data_path: str
    :param split: The dataset split to use. Default is `test`.
        Available choices are: `train`, `validation`, `test`.
    :type split: str
    """

    data_path: str = ""
    split: Annotated[str, Choices("train", "validation", "test")] = "test"


@QA_DATASETS("literary_qa", config_class=LiteraryQADatasetConfig)
@KNOWLEDGE_QA_DATASETS("literary_qa", config_class=LiteraryQADatasetConfig)
class LiteraryQADataset(KnowledgeQADatasetBase):
    def __init__(self, config: LiteraryQADatasetConfig):
        # load the dataset
        assert (
            config.data_path != ""
        ), "data_path must be specified for LiteraryQADataset."
        data_path = Path(config.data_path) / f"{config.split}.jsonl"
        reader = LineDelimitedReader(data_path)
        self._context_data = {}
        self._queries_data = {}
        self._qrels_data = {}
        self._answers_data = {}
        for row in reader:
            metadata = row.get("metadata", {})
            metadata["gutenberg_id"] = row["gutenberg_id"]
            context = Context(
                context_id=row["document_id"],
                data={
                    "title": row["title"],
                    "summary": row["summary"],
                    "text": row["text"],
                },
                source="literary_qa",
                meta_data=metadata,
            )
            self._context_data[context.context_id] = context
            for n, qa_pair in enumerate(row["qas"]):
                qid = f"{row['document_id']}_qa_{n}"
                self._queries_data[qid] = qa_pair["question"]
                self._qrels_data[qid] = {row["document_id"]: 1.0}
                self._answers_data[qid] = qa_pair["answers"]
        return

    @property
    def _queries(self) -> dict[str, str]:
        return self._queries_data

    @property
    def _qrels(self) -> dict[str, dict[str, float]]:
        return self._qrels_data

    @property
    def _contexts(self) -> dict[str, Context]:
        return self._context_data

    @property
    def _answers(self) -> dict[str, list[str]]:
        return self._answers_data
