from flexrag.common import configure
from flexrag.datasets.benchmarks import MTEBDataset, MTEBDatasetConfig
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
class MTEBRetrievalTaskConfig(RetrievalTaskConfig, MTEBDatasetConfig):
    """Configuration for MTEB Retrieval Task."""

    subset: str = "nq"
    split: str = "test"


@TASKS("mteb", config_class=MTEBRetrievalTaskConfig)
class MTEBRetrievalTask(RetrievalTask):
    """MTEB Retrieval Task."""

    def load_dataset(self) -> MTEBDataset:
        """Load the MTEB dataset for the Retrieval task.

        :return: The MTEB dataset for the Retrieval task.
        :rtype: MTEBDataset
        """
        return MTEBDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "ndcg": RetrievalNDCG(RetrievalNDCGConfig(k_values=[1, 3, 5, 10])),
            "recall": RetrievalRecall(RetrievalRecallConfig(k_values=[1, 3, 5, 10])),
            "map": RetrievalMAP(RetrievalMAPConfig(k_values=[1, 3, 5, 10])),
            "mrr": RetrievalMRR(),
        }
        return Evaluator(metrics)
