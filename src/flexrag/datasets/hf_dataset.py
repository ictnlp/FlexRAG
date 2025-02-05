from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset as _Dataset
from datasets import DatasetDict as _DatasetDict
from datasets import load_dataset

from .dataset import MappingDataset


@dataclass
class HFDatasetConfig:
    """The configuration for the ``HFDataset``.
    The ``HFDataset`` is a wrapper class that employs the ``load_dataset`` method in HuggingFace ``datasets`` library to load the dataset.

    :param path: Path or name of the dataset.
    :type path: str
    :param name: Defining the name of the dataset configuration.
    :type name: Optional[str]
    :param data_dir: Defining the ``data_dir`` of the dataset configuration.
    :type data_dir: Optional[str]
    :param data_files: Paths to source data files.
    :type data_files: list[str]
    :param split: Which split of the data to load.
    :type split: Optional[str]
    :param cache_dir: Directory to read/write data.
    :type cache_dir: Optional[str]
    :param token: Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
    :type token: Optional[str]
    :param trust_remote_code:  Whether or not to allow for datasets defined on the Hub using a dataset script.
    :type trust_remote_code: bool

    For example, you can load the dataset from the HuggingFace by running the following code:

        >>> cfg = HFDatasetConfig(
        ...     path="mteb/nq",
        ...     split="test",
        ... )
        >>> dataset = HFDataset(cfg)

    You can also load the dataset from a local repository by specifying the path:

        >>> cfg = HFDatasetConfig(
        ...     path="json",
        ...     data_files=["path/to/local/my_dataset.json"],
        ... )
        >>> dataset = HFDataset(cfg)

    For more information about the parameters, please refer to the HuggingFace ``datasets`` documentation:
    https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset
    """

    path: str
    name: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: list[str] = field(default_factory=list)
    split: Optional[str] = None
    cache_dir: Optional[str] = None
    token: Optional[str] = None
    trust_remote_code: bool = False


class HFDataset(MappingDataset):
    """HFDataset is a dataset that wraps the HaggingFace ``datasets`` library."""

    dataset: _Dataset

    def __init__(self, cfg: HFDatasetConfig) -> None:
        super().__init__()
        self.dataset = load_dataset(
            path=cfg.path,
            name=cfg.name,
            data_dir=cfg.data_dir,
            data_files=cfg.data_files if cfg.data_files else None,
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
