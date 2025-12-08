import csv
import json
import re
import sys
from collections.abc import Iterable, Iterator
from csv import reader as csv_reader
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Literal

from .file_mixin import FileReaderMixin

csv.field_size_limit(sys.maxsize)


class LineDelimitedReader(Iterable[dict], FileReaderMixin):
    """The iterative reader for loading line delimited file (csv, tsv, jsonl)."""

    def __init__(
        self,
        file_path: PathLike,
        titles: list[str] = None,
        data_range: list[int, int] = [0, -1],
        encoding: str = "utf-8",
        file_format: Literal["csv", "tsv", "jsonl"] | None = None,
        delimiter: str = None,
    ) -> None:
        """Initialize the LineDelimitedReader.

        :param file_path: The path to the line delimited file.
        :type file_path: PathLike
        :param titles: The field names of the corpus data.
            This option is only used when the corpus is in `tsv` or `csv` format.
            If not specified, the field names will be inferred from the first line of the file.
        :type titles: list[str]
        :param data_range: The data ranges to load from the file.
            If end_point is -1, it will read to the end of the file.
            If not specified, it will read the whole file.
        :type data_range: list[int, int]
        :param encoding: The encoding of the file.
        :type encoding: str
        :param file_format: The format of the file.
            Available choices are: `csv`, `tsv`, and `jsonl`.
            If not specified, it will detect the format automatically.
        :type file_format: Literal["csv", "tsv", "jsonl"] | None
        :param delimiter: The delimiter for csv/tsv files.
            If not specified, it will use tab for tsv and comma for csv.
            If a single character is provided, it will be used as the delimiter.
            If multiple characters are provided, it will be treated as a regex pattern.
        :type delimiter: str

        Example 1: Loading specific lines from a file.

            >>> reader = LineDelimitedReader(
            ...     file_path="data.jsonl",
            ...     data_range=[10, 10],
            ...     encoding="utf-8",
            ... )
            >>> items = [i for i in reader]

        Example 2: Loading the tsv file with given titles.

            >>> reader = LineDelimitedReader(
            ...     file_path="data.tsv",
            ...     titles=["id", "url", "title", "text"],
            ...     encoding="utf-8",
            ... )
            >>> items = [i for i in reader]
        """
        self.file_path = Path(file_path)
        self.data_range = data_range
        self.encoding = encoding
        self.titles = titles
        self._file_format = file_format
        self.delimiter = delimiter
        return

    def __iter__(self) -> Iterator[dict]:
        # read data
        start_point, end_point = self.data_range
        if end_point > 0:
            assert end_point > start_point, f"Invalid data range: {self.data_range}"
        match self.format:
            case fmt if fmt in {"jsonl"}:
                with self._open_file(self.file_path, "r", encoding=self.encoding) as f:
                    for i, line in enumerate(f):
                        if i < start_point:
                            continue
                        if (end_point > 0) and (i >= end_point):
                            break
                        yield json.loads(line)
            case fmt if fmt in {"tsv", "csv"}:
                if self.titles is not None and len(self.titles) > 0:
                    title = self.titles
                else:
                    title = []
                    start_point -= 1  # skip the header row
                    end_point -= 1  # skip the header row
                # prepare delimiter
                if self.delimiter is None:
                    delimiter = "\t" if fmt == "tsv" else ","
                elif len(self.delimiter) == 1:
                    delimiter = self.delimiter
                else:
                    delimiter = re.compile(self.delimiter)
                # parse file
                with self._open_file(self.file_path, "r", encoding=self.encoding) as f:
                    if not isinstance(delimiter, re.Pattern):
                        for i, row in enumerate(csv_reader(f, delimiter=delimiter)):
                            if (i == 0) and (len(title) == 0):
                                title = row
                                continue
                            if i < start_point:
                                continue
                            if (end_point > 0) and (i >= end_point):
                                break
                            yield dict(zip(title, row))
                    else:
                        for i, line in enumerate(f):
                            if (i == 0) and (len(title) == 0):
                                title = re.split(delimiter, line.strip())
                                continue
                            if i < start_point:
                                continue
                            if (end_point > 0) and (i >= end_point):
                                break
                            row = re.split(delimiter, line.strip())
                            yield dict(zip(title, row))
            case _:
                raise ValueError(f"Unsupported file format: {self.file_path}")
        return

    def __repr__(self) -> str:
        return f"LineDelimitedReader({self.file_path})"

    @cached_property
    def format(self) -> str:
        """Detect the format of the given file."""
        # 1. Check if the format is already given
        if self._file_format is not None:
            return self._file_format

        # 2. Detect the format by the suffix
        suffix = self.file_path.suffix.lower().lstrip(".")
        if suffix in {"csv", "tsv", "jsonl"}:
            return suffix

        # 3. Detect the format by the first non-blank line
        with self._open_file(self.file_path, "r", encoding=self.encoding) as f:
            for line in f:
                if not line.strip():
                    continue
                line = line.strip()
                try:
                    json.loads(line)
                    return "jsonl"
                except json.JSONDecodeError:
                    pass
                if "\t" in line:
                    return "tsv"
                if "," in line:
                    return "csv"
                break
        raise ValueError("Unable to detect file format")
