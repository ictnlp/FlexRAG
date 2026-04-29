from typing import Annotated

from flexrag.common import Choices, configure
from flexrag.datasets.benchmarks import MSMARCODataset, MSMARCODatasetConfig
from flexrag.metrics import (
    Evaluator,
    RetrievalMAP,
    RetrievalMAPConfig,
    RetrievalMRR,
    RetrievalNDCG,
    RetrievalNDCGConfig,
    RetrievalRecall,
    RetrievalRecallConfig,
)

from ..retrieval_base import RetrievalTask, RetrievalTaskConfig
from ..task_base import TASKS


@configure
class MSMARCORetrievalTaskConfig(RetrievalTaskConfig, MSMARCODatasetConfig):
    """Configuration for MSMARCO Retrieval Task."""

    subset: Annotated[
        str,
        Choices(
            "msmarco_passage_ranking_v1",
            "msmarco_passage_ranking_v2",
            "msmarco_document_ranking_v1",
            "msmarco_document_ranking_v2",
        ),
    ] = "msmarco_passage_ranking_v1"
    split: str = "dev"


@TASKS("ms_marco", config_class=MSMARCORetrievalTaskConfig)
class MSMARCORetrievalTask(RetrievalTask):
    """MSMARCO Retrieval Task."""

    def load_dataset(self) -> MSMARCODataset:
        """Load the MSMARCO dataset for the Retrieval task.

        :return: The MSMARCO dataset for the Retrieval task.
        :rtype: MSMARCODataset
        """
        return MSMARCODataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "ndcg": RetrievalNDCG(RetrievalNDCGConfig(k_values=[1, 3, 5, 10])),
            "recall": RetrievalRecall(RetrievalRecallConfig(k_values=[1, 3, 5, 10])),
            "map": RetrievalMAP(RetrievalMAPConfig(k_values=[1, 3, 5, 10])),
            "mrr": RetrievalMRR(),
        }
        return Evaluator(metrics)
