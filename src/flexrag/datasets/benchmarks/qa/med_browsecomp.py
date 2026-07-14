from pathlib import Path
from typing import Literal, Optional

from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure

from ...core import DATASETS, MappingDataset, QASample
from ...reader import LineDelimitedReader


@configure
class MedBrowseCompDatasetConfig:
    """Configuration for MedBrowseComp dataset.

    `MedBrowseComp <https://arxiv.org/abs/2505.14963>`_
    is a benchmark of 1,000+ human-curated clinical questions that evaluates whether
    agentic LLMs can reliably retrieve and synthesize up-to-date, multi-hop medical
    evidence from live, domain-specific sources.

    :param subset: The subset of the MedBrowseComp dataset to use.
        Options are "50", "605", and "cua". Default is "50".
    :type subset: Literal["50", "605", "cua"]
    :param data_path: The path to the MedBrowseComp dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    subset: Literal["50", "605", "cua"] = "50"
    data_path: Optional[str] = None


@DATASETS("med_browsecomp", config_class=MedBrowseCompDatasetConfig)
class MedBrowseCompDataset(MappingDataset[QASample]):
    """Dataset for MedBrowseComp benchmark."""

    def __init__(self, config: MedBrowseCompDatasetConfig):
        self._subset = config.subset
        if config.data_path is not None:
            data_path = Path(config.data_path)
        else:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "med_browsecomp"

        # download the dataset if not exists
        if not data_path.exists():
            snapshot_download(
                repo_id="AIM-Harvard/MedBrowseComp",
                repo_type="dataset",
                local_dir=data_path.as_posix(),
            )

        # load the dataset
        match self._subset:
            case "50":
                data_path = data_path / "MedBrowseComp_50.csv"
            case "605":
                data_path = data_path / "MedBrowseComp_605.csv"
            case "cua":
                data_path = data_path / "MedBrowseComp_CUA.csv"
            case _:
                raise ValueError(f"Invalid subset: {self._subset}")
        self._data = list(LineDelimitedReader(data_path))
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> QASample:
        item = self._data[index]
        return QASample(
            question=item["prompt"],
            answers=[item["gold"]],
            metadata={"task_name": item["task_name"]},
        )
