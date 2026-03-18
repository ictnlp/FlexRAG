import base64
import hashlib
from pathlib import Path
from typing import Literal, Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure

from ...core import DATASETS, MappingDataset, QASample


@configure
class DeepResearch9KDatasetConfig:
    """Configuration for DeepResearch9K dataset.

    `DeepResearch9K <https://arxiv.org/abs/2603.01152>`_
    is a large-scale benchmark of 9,000 multi-level deep-research questions, paired
    with high-quality search trajectories and verifiable answers, designed to support
    realistic agent evaluation and training.

    :param data_path: The path to the DeepResearch9K dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


@DATASETS("deepresearch_9k", config_class=DeepResearch9KDatasetConfig)
class DeepResearch9KDataset(MappingDataset[QASample]):
    """Dataset for DeepResearch9K benchmark."""

    def __init__(self, config: DeepResearch9KDatasetConfig):
        if config.data_path is not None:
            data_path = Path(config.data_path)
        else:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "deepresearch_9k"

        # download the dataset if not exists
        if not data_path.exists():
            snapshot_download(
                repo_id="artillerywu/DeepResearch-9K",
                repo_type="dataset",
                local_dir=data_path.as_posix(),
            )

        # load the dataset
        self._raw_data = load_dataset(data_path.as_posix(), split="train")
        return

    def __len__(self) -> int:
        return len(self._raw_data)

    def get_item(self, index: int) -> QASample:
        item = self._raw_data[index]
        return QASample(
            question=item["question"],
            answers=[item["final answer"]],
            meta_data={
                "difficulty": item["difficulty"],
                "search trajectory": item["search trajectory"],
            },
        )
