from typing import Optional

from flexrag.common import FLEXRAG_CACHE_DIR, configure, download

from ...core import MappingDataset, QASample
from ...reader import LineDelimitedReader

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


class SimpleQADataset(MappingDataset[QASample]):
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
        self._metadata = {}
        for idx, row in enumerate(reader):
            self._queries_data[str(idx)] = row["problem"]
            self._answers_data[str(idx)] = [row["answer"]]
            self._metadata[str(idx)] = eval(row["metadata"])
        self._qids = list(self._queries_data.keys())
        return

    def __len__(self) -> int:
        return len(self._queries_data)

    def get_item(self, index: int) -> QASample:
        qid = self._qids[index]
        return QASample(
            question=self._queries_data[qid],
            answers=self._answers_data[qid],
            metadata=self._metadata[qid],
        )
