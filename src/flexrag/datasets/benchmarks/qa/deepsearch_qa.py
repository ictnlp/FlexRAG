import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure

from ...core import DATASETS, MappingDataset, QASample
from ...reader import LineDelimitedReader


@configure
class DeepSearchQADatasetConfig:
    """Configuration for DeepSearch QA dataset.

    `DeepSearch QA <https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf>`_
    is a 900-prompt, open-web benchmark designed to evaluate agents' ability to
    perform long-horizon, multi-step information seeking, requiring systematic
    evidence aggregation, entity resolution, and principled stopping to produce
    exhaustive and precise answer sets across diverse domains.

    :param data_path: The path to the DeepSearch QA dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


@DATASETS("deepsearch_qa", config_class=DeepSearchQADatasetConfig)
class DeepSearchQADataset(MappingDataset[QASample]):
    """Dataset for DeepSearch QA benchmark."""

    def __init__(self, config: DeepSearchQADatasetConfig):
        if config.data_path is not None:
            data_path = Path(config.data_path)
        else:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "deepsearch_qa"

        # download the dataset if not exists
        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="google/deepsearchqa",
                repo_type="dataset",
                local_dir=data_path.as_posix(),
                token=os.getenv("HF_TOKEN"),
            )

        self._data = list(LineDelimitedReader(data_path / "DSQA-full.csv"))
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> QASample:
        item = self._data[index]
        return QASample(
            question=item["problem"],
            answers=[item["answer"]],
            meta_data={
                "problem_category": item["problem_category"],
                "answer_type": item["answer_type"],
            },
        )
