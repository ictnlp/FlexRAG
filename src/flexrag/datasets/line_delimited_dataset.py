import json
from csv import reader as csv_reader
from dataclasses import dataclass, field
from glob import glob
from typing import Iterator

from .dataset import IterableDataset


@dataclass
class LineDelimitedDatasetConfig:
    """The configuration for ``LineDelimitedDataset``.

    :param file_paths: The paths to the line delimited files.
        It supports unix style path pattern.
    :type file_paths: list[str]
    :param data_ranges: The data ranges to load from the files.
        The format is a list of [start_point, end_point] for each file.
        If end_point is -1, it will read to the end of the file.
        If not specified, it will read the whole file.
    :type data_ranges: list[list[int, int]]
    :param encoding: The encoding of the files.
    :type encoding: str

    Example 1: Loading specific lines from given files.

        >>> cfg = LineDelimitedDatasetConfig(
        ...     file_paths=["data1.jsonl", "data2.csv"],
        ...     data_ranges=[[0, 10], [0, 20]],
        ...     encoding="utf-8",
        ... )
        >>> dataset = LineDelimitedDataset(cfg)
        >>> items = [i for i in dataset]

    Example 2: Loading multiple files using unix style path pattern.

        >>> cfg = LineDelimitedDatasetConfig(
        ...     file_paths=["data/*.jsonl"],
        ...     encoding="utf-8",
        ... )
        >>> dataset = LineDelimitedDataset(cfg)
        >>> items = [i for i in dataset]
    """

    file_paths: list[str]
    data_ranges: list[list[int, int]] = field(default_factory=list)
    encoding: str = "utf-8"


class LineDelimitedDataset(IterableDataset):
    """The iterative dataset for loading line delimited files (csv, tsv, jsonl)."""

    def __init__(self, cfg: LineDelimitedDatasetConfig) -> None:
        # process unix style path
        file_paths = [glob(p) for p in cfg.file_paths]
        for p in file_paths:
            if len(p) != 1:
                assert (
                    len(cfg.data_ranges) == 0
                ), "`data_ranges` do not support unix style path pattern"
        file_paths = [p for file_path in file_paths for p in file_path]

        # check data_ranges consistency
        if len(cfg.data_ranges) != 0:
            assert len(cfg.data_ranges) == len(file_paths), "Invalid data ranges"
        else:
            cfg.data_ranges = [[0, -1] for _ in file_paths]

        self.file_paths = file_paths
        self.data_ranges = cfg.data_ranges
        self.encoding = cfg.encoding
        return

    def __iter__(self) -> Iterator[dict]:
        # read data
        for file_path, data_range in zip(self.file_paths, self.data_ranges):
            start_point, end_point = data_range
            if end_point > 0:
                assert end_point > start_point, f"Invalid data range: {data_range}"
            if file_path.endswith(".jsonl"):
                with open(file_path, "r", encoding=self.encoding) as f:
                    for i, line in enumerate(f):
                        if i < start_point:
                            continue
                        if (end_point > 0) and (i >= end_point):
                            break
                        yield json.loads(line)
            elif file_path.endswith(".tsv"):
                title = []
                with open(file_path, "r", encoding=self.encoding) as f:
                    for i, row in enumerate(csv_reader(f, delimiter="\t")):
                        if i == 0:
                            title = row
                            continue
                        if i <= start_point:
                            continue
                        if (end_point > 0) and (i > end_point):
                            break
                        yield dict(zip(title, row))
            elif file_path.endswith(".csv"):
                title = []
                with open(file_path, "r", encoding=self.encoding) as f:
                    for i, row in enumerate(csv_reader(f)):
                        if i == 0:
                            title = row
                            continue
                        if i <= start_point:
                            continue
                        if (end_point > 0) and (i > end_point):
                            break
                        yield dict(zip(title, row))
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        return

    def __repr__(self) -> str:
        return f"LineDelimitedDataset({self.file_paths})"
