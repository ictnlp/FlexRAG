import logging
import os
from dataclasses import dataclass

import numpy as np

from .index_base import DenseIndex, DenseIndexConfig


logger = logging.getLogger(__name__)


@dataclass
class ScaNNIndexConfig(DenseIndexConfig):
    num_leaves: int = 2000
    num_leaves_to_search: int = 500
    num_neighbors: int = 10
    anisotropic_quantization_threshold: float = 0.2


class ScaNNIndex(DenseIndex):
    def __init__(self, cfg: ScaNNIndexConfig) -> None:
        super().__init__(cfg)
        # check scann
        try:
            import scann

            self.scann = scann
        except:
            raise ImportError("Please install scann by running `pip install scann`")

        # preapre basic args
        self.train_num = self.index_train_num
        self.log_interval = cfg.log_interval
        self.index_path = self.index_path

        # prepare index
        if self.index_path is not None and os.path.exists(self.index_path):
            self.deserialize()
        else:
            self.index = self.prepare_index(
                num_leaves=cfg.num_leaves,
                num_leaves_to_search=cfg.num_leaves_to_search,
                num_neighbors=cfg.num_neighbors,
                anisotropic_quantization_threshold=cfg.anisotropic_quantization_threshold,
                distance_func=self.distance_function,
            )
        return

    def prepare_index(
        self,
        num_leaves: int,
        num_leaves_to_search: int,
        num_neighbors: int,
        anisotropic_quantization_threshold: float,
        distance_func: str = "IP",
    ):
        distance_measure = "dot_product" if distance_func == "IP" else "squared_l2"
        return (
            self.scann.scann_ops_pybind.builder(
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
        self.index = self.scann.scann_ops_pybind.load_searcher(self.index_path)
        return

    # TODO: implement clear method
    def clear(self):
        raise NotImplementedError

    @property
    def is_trained(self):
        return not isinstance(self.index, self.scann.ScannBuilder)

    @property
    def embedding_size(self):
        return self.index.num_columns

    def __len__(self) -> int:
        if isinstance(self.index, self.scann.ScannBuilder):
            return 0
        return self.index.size()
