from pathlib import Path
from typing import Optional

from flexrag.common import FLEXRAG_CACHE_DIR, configure
from flexrag.common.dataclasses import Context
from flexrag.common.logging import LOGGER_MANAGER
from flexrag.common.misc import download_and_extract

from ..reader import LineDelimitedReader
from .qa_dataset_base import KNOWLEDGE_QA_DATASETS, QA_DATASETS, KnowledgeQADatasetBase

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.attribute_qa")


@configure
class AttributedQADatasetConfig:
    """Configuration for AttributedQADataset.

    `AttributedQA <https://arxiv.org/abs/2212.08037>`_ designed to evaluate how
    well large language models can ground their answers in verifiable sources,
    providing a benchmark for measuring and improving attribution in
    information-seeking tasks.

    :param data_path: The path to the AttributedQA dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


RESOURCES = {
    "corpus": "https://storage.googleapis.com/gresearch/attributed_language_models/wikipedia.zip",
    "dataset": "https://github.com/google-research-datasets/Attributed-QA/raw/refs/heads/main/ratings.zip",
}


@QA_DATASETS("attributed_qa", config_class=AttributedQADatasetConfig)
@KNOWLEDGE_QA_DATASETS("attributed_qa", config_class=AttributedQADatasetConfig)
class AttributedQADataset(KnowledgeQADatasetBase):
    def __init__(self, config: AttributedQADatasetConfig):
        # Download the dataset if not exists
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "attributed_qa"
        else:
            data_path = Path(config.data_path)
        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            download_and_extract(RESOURCES["dataset"], data_path.as_posix())

        # Download the corpus if not exists
        corpus_path = data_path / "wikipedia"
        if not corpus_path.exists():
            download_and_extract(RESOURCES["corpus"], corpus_path.as_posix())

        # Load the corpus
        self._context_data = {}
        logger.info("Loading AttributedQA corpus...")
        for file_path in corpus_path.iterdir():
            reader = LineDelimitedReader(file_path)
            for line in reader:
                context_id = line["id"]
                self._context_data[context_id] = Context(
                    context_id=context_id,
                    data={"text": line["contents"]},
                    source="wikipedia_2021",
                )
        logger.info(
            f"Loaded {len(self._context_data)} contexts from AttributedQA corpus."
        )

        # Load the dataset
        self._queries_data = {}
        self._answers_data = {}
        self._qrels_data = {}
        self._meta_data = {}
        dataset_file = data_path / "ratings.csv"
        reader = LineDelimitedReader(dataset_file)
        for line in reader:
            qid = line[""]
            self._queries_data[qid] = line["question"]
            self._answers_data[qid] = [line["answer"]]
            context_id = line["attribution"]
            self._qrels_data[qid] = {context_id: 1.0}
            self._meta_data[qid] = {
                "nli_score": float(line["nli_score"]),
                "human_rating": line["human_rating"],
                "auto_ais": line["auto_ais"],
                "system_name": line["system_name"],
            }
        return

    @property
    def _queries(self) -> dict[str, str]:
        return self._queries_data

    @property
    def _answers(self) -> dict[str, list[str]] | None:
        return self._answers_data

    @property
    def _qrels(self) -> dict[str, dict[str, float]]:
        return self._qrels_data

    @property
    def _contexts(self) -> dict[str, Context]:
        return self._context_data
