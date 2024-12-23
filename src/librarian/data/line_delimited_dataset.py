from typing import Optional, Iterator
from glob import glob
from itertools import zip_longest
import json
from csv import reader as csv_reader

from .dataset import Dataset


class LineDelimitedDataset(Dataset):
    def __init__(
        self,
        file_paths: list[str] | str,
        data_ranges: Optional[list[list[int, int]] | list[int, int]] = None,
    ):
        # for single file path
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            if data_ranges is not None:
                assert isinstance(data_ranges[0], int), "Invalid data ranges"
                assert isinstance(data_ranges[1], int), "Invalid data ranges"
                data_ranges = [data_ranges]

        # process unix style path
        file_paths = [glob(p) for p in file_paths]
        for p in file_paths:
            if len(p) != 1:
                assert (
                    data_ranges is None
                ), "Data ranges do not support unix style path pattern"
        file_paths = [p for file_path in file_paths for p in file_path]

        if data_ranges is None:
            data_ranges = []
        else:
            assert len(data_ranges) == len(file_paths), "Invalid data ranges"

        self.file_paths = file_paths
        self.data_ranges = data_ranges
        return

    def __iter__(self) -> Iterator[dict]:
        # read data
        for file_path, data_range in zip_longest(
            self.file_paths, self.data_ranges, fillvalue=[0, -1]
        ):
            start_point, end_point = data_range
            if end_point > 0:
                assert end_point > start_point, f"Invalid data range: {data_range}"
            if file_path.endswith(".jsonl"):
                with open(file_path, "r") as f:
                    for i, line in enumerate(f):
                        if i < start_point:
                            continue
                        if (end_point > 0) and (i >= end_point):
                            break
                        yield json.loads(line)
            elif file_path.endswith(".tsv"):
                title = []
                with open(file_path, "r") as f:
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
                with open(file_path, "r") as f:
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
