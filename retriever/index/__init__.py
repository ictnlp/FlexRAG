from argparse import ArgumentParser, Namespace

from .index_base import DenseIndex
from .faiss_index import FaissIndex
from .scann_index import ScaNNIndex

__all__ = ["FaissIndex", "ScaNNIndex", "DenseIndex"]


def load_index(
    index_args: Namespace,
    index_path: str,
    log_interval: int = 100,
    embedding_size: int = 768,
    device_id: int = -1,
) -> DenseIndex:
    # prepare args
    index_args = vars(index_args)
    index_args["log_interval"] = log_interval
    index_args["device_id"] = device_id
    index_args["index_path"] = index_path
    index_args["embedding_size"] = embedding_size,
    # load index
    if index_args["index_type"] == "faiss":
        index = FaissIndex(**index_args)
    elif index_args["index_type"] == "scann":
        index = ScaNNIndex(**index_args)
    else:
        raise ValueError(f"Unsupported index type: {index_args.index_type}")
    return index


def add_index_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--index_type",
        type=str,
        default="faiss",
        choices=["faiss", "scann"],
        help="Index type",
    )
    parser.add_argument(
        "--index_train_num",
        type=int,
        default=1000000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--distance_func",
        type=str,
        default="IP",
        choices=["IP", "L2"],
        help="Distance function for index",
    )
    parser = FaissIndex.add_args(parser)
    parser = ScaNNIndex.add_args(parser)
    return parser
