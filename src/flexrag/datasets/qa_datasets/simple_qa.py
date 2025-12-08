from typing import Mapping, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, configure, download

from ..reader import LineDelimitedReader
from .qa_dataset_base import QA_DATASETS, QADatasetBase

RESOURCE_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
)


@configure
class SimpleQADatasetConfig:
    """Configuration for SimpleQADataset.

    `SimpleQA <https://arxiv.org/abs/2411.04368>`_ is a benchmark dataset
    released by OpenAI designed to measure the factual accuracy of language models.
    It consists of short, fact-seeking questions with clear, verifiable answers,
    aiming to evaluate a model's ability to provide correct information or
    refuse to answer when unsure, rather than hallucinating.

    :param data_path: The path to the SimpleQA dataset file. If not provided,
        the dataset will be downloaded automatically to FLEXRAG_CACHE_DIR.
        Default: None.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


@QA_DATASETS("simple_qa", config_class=SimpleQADatasetConfig)
class SimpleQADataset(QADatasetBase):
    def __init__(self, config: SimpleQADatasetConfig):
        # Download the dataset if not already present
        if config.data_path is not None:
            data_path = config.data_path
        else:
            data_path = (
                FLEXRAG_CACHE_DIR / "datasets" / "simple_qa" / "simple_qa_test_set.csv"
            )
        if not data_path.exists():
            download(RESOURCE_URL, data_path)

        # load the dataset
        reader = LineDelimitedReader(data_path)
        self._answers_data = {}
        self._queries_data = {}
        self._meta_data = {}
        for idx, row in enumerate(reader):
            self._queries_data[str(idx)] = row["problem"]
            self._answers_data[str(idx)] = [row["answer"]]
            self._meta_data[str(idx)] = eval(row["metadata"])
        return

    @property
    def _queries(self) -> Mapping[str, str]:
        return self._queries_data

    @property
    def _answers(self) -> Mapping[str, list[str]] | None:
        return self._answers_data

    def get_item(self, index: int):
        data = super().get_item(index)
        data.meta_data = self._meta_data[str(index)]
        return data
