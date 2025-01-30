from dataclasses import dataclass
from typing import Optional

from datasets import Dataset as _Dataset
from datasets import DatasetDict as _DatasetDict
from datasets import load_dataset

from .dataset import MappingDataset


@dataclass
class HFDatasetConfig:
    """The configuration for the ``HFDataset``.

    :param path: Path or name of the dataset.
    :type path: str
    :param name: Defining the name of the dataset configuration.
    :type name: Optional[str]
    :param data_dir: Defining the ``data_dir`` of the dataset configuration.
    :type data_dir: Optional[str]
    :param data_files: Path(s) to source data file(s).
    :type data_files: Optional[str]
    :param split: Which split of the data to load.
    :type split: Optional[str]
    :param cache_dir: Directory to read/write data.
    :type cache_dir: Optional[str]
    :param token: Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
    :type token: Optional[str]
    :param trust_remote_code:  Whether or not to allow for datasets defined on the Hub using a dataset script.
    :type trust_remote_code: bool

    For more information, please refer to the HuggingFace ``datasets`` documentation:
    https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset
    """

    path: str
    name: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[str] = None
    split: Optional[str] = None
    cache_dir: Optional[str] = None
    token: Optional[str] = None
    trust_remote_code: bool = False


class HFDataset(MappingDataset):
    """HFDataset is a dataset that wraps the HaggingFace ``datasets`` library."""

    dataset: _Dataset

    def __init__(self, cfg: HFDatasetConfig) -> None:
        self.dataset = load_dataset(
            path=cfg.path,
            name=cfg.name,
            data_dir=cfg.data_dir,
            data_files=cfg.data_files,
            split=cfg.split,
            cache_dir=cfg.cache_dir,
            token=cfg.token,
            trust_remote_code=cfg.trust_remote_code,
        )
        if isinstance(self.dataset, _DatasetDict):
            raise ValueError(
                "Split is missing.\n"
                "Please pick one among the following splits: "
                f"{list(self.dataset.keys())}"
            )
        return

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
