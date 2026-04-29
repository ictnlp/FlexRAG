import os
from typing import Annotated, Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, configure

from ...core import DATASETS, MappingDataset, QASample


@configure
class GAIADatasetConfig:
    """Configuration for GAIA dataset.

    `GAIA <https://arxiv.org/abs/2311.12983>`_ is a real-world benchmark
    of 466 human-easy yet AI-hard questions that evaluates general AI
    assistants on core abilities such as reasoning, multimodality, web
    browsing, and tool use, highlighting a large robustness gap between
    humans and current models.

    :param data_path: The path to the GAIA dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param subset: The subset of GAIA to use. Default is `2023_all`.
        Available choices are: `2023_all`, `2023_level1`, `2023_level2`, and `2023_level3`.
    :type subset: str
    :param split: The split of the dataset to use. Default is `validation`.
        Available choices are: `test`, and `validation`.
    :type split: str
    """

    data_path: Optional[str] = None
    subset: Annotated[
        str,
        Choices(
            "2023_all",
            "2023_level1",
            "2023_level2",
            "2023_level3",
        ),
    ] = "2023_all"
    split: Annotated[str, Choices("test", "validation")] = "validation"


@DATASETS("gaia", config_class=GAIADatasetConfig)
class GAIADataset(MappingDataset[QASample]):
    """Dataset for GAIA benchmark."""

    def __init__(self, config: GAIADatasetConfig):
        if config.data_path is not None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "gaia"
        else:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "gaia"

        # download the dataset if not exists
        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir=data_path.as_posix(),
                token=os.getenv("HF_TOKEN"),
            )

        # load dataset
        self._data = load_dataset(
            data_path.as_posix(),
            name=config.subset,
            split=config.split,
        )
        self._data_path = data_path
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int):
        item = self._data[index]
        if item["file_path"] != "":
            file_path = (self._data_path / item["file_path"]).as_posix()
        else:
            file_path = ""
        data = QASample(
            question_id=item["task_id"],
            question=item["Question"],
            answers=[item["Final answer"]],
            meta_data={
                "level": item["Level"],
                "annotator_metadata": item["Annotator Metadata"],
                "file_name": item["file_name"],
                "file_path": file_path,
            },
        )
        return data
