from flexrag.common import configure
from flexrag.datasets.benchmarks import (
    MultiLongDocRetrievalDataset,
    MultiLongDocRetrievalDatasetConfig,
)
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
class MLDRRetrievalTaskConfig(RetrievalTaskConfig, MultiLongDocRetrievalDatasetConfig):
    """Configuration for MLDR Retrieval Task."""

    split: str = "dev"


@TASKS("mldr", config_class=MLDRRetrievalTaskConfig)
class MLDRRetrievalTask(RetrievalTask):
    """MLDR Retrieval Task."""

    def load_dataset(self) -> MultiLongDocRetrievalDataset:
        """Load the MLDR dataset for the Retrieval task.

        :return: The MLDR dataset for the Retrieval task.
        :rtype: MultiLongDocRetrievalDataset
        """
        return MultiLongDocRetrievalDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "ndcg": RetrievalNDCG(RetrievalNDCGConfig(k_values=[1, 3, 5, 10])),
            "recall": RetrievalRecall(RetrievalRecallConfig(k_values=[1, 3, 5, 10])),
            "map": RetrievalMAP(RetrievalMAPConfig(k_values=[1, 3, 5, 10])),
            "mrr": RetrievalMRR(),
        }
        return Evaluator(metrics)
