import json
import pathlib
import sys
from argparse import ArgumentParser
from csv import reader
from typing import Iterable

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


from kylin.retriever import BM25Retriever, LocalRetriever, Retriever


def read_data(file_paths: list[str]) -> Iterable:
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


if __name__ == "__main__":
    # parse args
    parser = ArgumentParser()
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="The interval to log the information",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        nargs="+",
        help="The path to the corpus file",
    )
    parser = Retriever.add_args(parser)
    parser = LocalRetriever.add_args(parser)
    parser = BM25Retriever.add_args(parser)
    args = parser.parse_args()

    # build retriever
    retriever = BM25Retriever(args)
    retriever.add_passages(passages=read_data(args.corpus_path), reinit=True)
