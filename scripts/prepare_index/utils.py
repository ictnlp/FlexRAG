import json
from csv import reader
from typing import Iterator


def read_data(file_paths: list[str]) -> Iterator:
    for file_path in file_paths:
        if file_path.endswith(".jsonl"):
            with open(file_path, "r") as f:
                for line in f:
                    yield json.loads(line)
        elif file_path.endswith(".tsv"):
            title = []
            with open(file_path, "r") as f:
                for i, row in enumerate(reader(f, delimiter="\t")):
                    if i == 0:
                        title = row
                    else:
                        yield dict(zip(title, row))
        elif file_path.endswith(".csv"):
            title = []
            with open(file_path, "r") as f:
                for i, row in enumerate(reader(f)):
                    if i == 0:
                        title = row
                    else:
                        yield dict(zip(title, row))
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
