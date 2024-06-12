import logging
import os
from argparse import ArgumentParser

import scann
import numpy as np

from .index_base import DenseIndex


logger = logging.getLogger(__name__)


class ScaNNIndex(DenseIndex):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--num_leaves",
            type=int,
            default=2000,
            help="Number of leaves in the tree",
        )
        parser.add_argument(
            "--num_leaves_to_search",
            type=int,
            default=500,
            help="Number of leaves to search",
        )
        parser.add_argument(
            "--num_neighbors",
            type=int,
            default=10,
            help="Number of neighbors to return",
        )
        parser.add_argument(
            "--anisotropic_quantization_threshold",
            type=float,
            default=0.2,
            help="Anisotropic quantization threshold",
        )
        return parser

    def __init__(
        self,
        index_path: str,
        num_leaves: int = 0,
        num_leaves_to_search: int = 8,
        num_neighbors: int = 1000,
        anisotropic_quantization_threshold: float = 0.2,
        distance_func: str = "IP",
        index_train_num: int = 1000000,
        log_interval: int = 100,
        **kwargs,
    ) -> None:
        # preapre basic args
        self.train_num = index_train_num
        self.log_interval = log_interval
        self.index_path = os.path.join(index_path, "scann_index")

        # prepare index
        if self.index_path is not None and os.path.exists(self.index_path):
            self.deserialize()
        else:
            self.index = self.prepare_index(
                num_leaves=num_leaves,
                num_leaves_to_search=num_leaves_to_search,
                num_neighbors=num_neighbors,
                anisotropic_quantization_threshold=anisotropic_quantization_threshold,
                distance_func=distance_func,
            )
        return

    def prepare_index(
        self,
        num_leaves: int,
        num_leaves_to_search: int,
        num_neighbors: int,
        anisotropic_quantization_threshold: float,
        distance_func: str = "IP",
    ) -> scann.ScannBuilder:
        distance_measure = "dot_product" if distance_func == "IP" else "squared_l2"
        return (
            scann.scann_ops_pybind.builder(
                None,
                num_neighbors,
                distance_measure=distance_measure,
            )
            .tree(
                num_leaves=num_leaves,
                num_leaves_to_search=num_leaves_to_search,
                training_sample_size=2500000,
            )
            .score_ah(
                dimensions_per_block=2,
                anisotropic_quantization_threshold=anisotropic_quantization_threshold,
            )
            .reorder(200)
        )

    def train_index(self, embeddings: np.ndarray) -> None:
        self.index.db = embeddings
        self.index = self.index.build()
        return

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray | list[int] = None,
        batch_size: int = 512,
    ) -> None:
        for idx in range(0, len(embeddings), batch_size):
            if (idx // batch_size) % self.log_interval == 0:
                logger.info(f"Adding {idx} / {len(embeddings)} embeddings.")
            embeds_to_add = embeddings[idx : idx + batch_size]
            if ids is not None:
                ids_to_add = ids[idx : idx + batch_size]
            else:
                ids_to_add = None
            self.add_embeddings_batch(embeds_to_add, ids_to_add)
        return

    def add_embeddings_batch(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray | list[int] = None,
    ) -> None:
        embeddings = embeddings.astype("float32")
        assert self.index.is_trained, "Index should be trained first"
        if ids is None:
            ids = np.arange(self.index.docids)
        self.index.upsert(database=embeddings)
        return

    def search(
        self,
        query_vectors: np.array,
        top_docs: int,
        batch_size: int = 512,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = query_vectors.astype("float32")
        scores = []
        indices = []
        for idx in range(0, len(query_vectors), batch_size):
            if (idx // batch_size) % self.log_interval == 0:
                logger.info(f"Searching {idx} / {len(query_vectors)} queries.")
            query = query_vectors[idx : idx + batch_size]
            r = self.search_batch(query, top_docs, **search_kwargs)
            scores.append(r[0])
            indices.append(r[1])
        scores = np.concatenate(scores, axis=0)
        indices = np.concatenate(indices, axis=0)
        return indices, scores

    def search_batch(
        self,
        query_vectors: np.array,
        top_docs: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = query_vectors.astype("float32")
        scores, indices = self.index.search_batched(
            query_vectors, top_docs, **search_kwargs
        )
        return indices, scores

    def serialize(self) -> None:
        logger.info(f"Serializing index to {self.index_path}")
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
        self.index.serialize(self.index_path)
        return

    def deserialize(self) -> None:
        logger.info(f"Loading index from {self.index_path}")
        self.index = scann.scann_ops_pybind.load_searcher(self.index_path)
        return

    @property
    def is_trained(self):
        return not isinstance(self.index, scann.ScannBuilder)

    def __len__(self) -> int:
        if isinstance(self.index, scann.ScannBuilder):
            return 0
        return self.index.docids
