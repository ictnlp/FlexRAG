import logging
import os
import shutil
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
    threads: int = 0


class ScaNNIndex(DenseIndex):
    def __init__(self, index_path: str, cfg: ScaNNIndexConfig) -> None:
        super().__init__(index_path, cfg)
        # check scann
        try:
            import scann

            self.scann = scann
        except:
            raise ImportError("Please install scann by running `pip install scann`")

        # set basic args
        self.num_leaves = cfg.num_leaves
        self.num_leaves_to_search = cfg.num_leaves_to_search
        self.num_neighbors = cfg.num_neighbors
        self.anisotropic_quantization_threshold = cfg.anisotropic_quantization_threshold
        self.threads = cfg.threads

        # prepare index
        if os.path.exists(self.index_path):
            self.deserialize()
        else:
            self.index = self._prepare_index()
        return

    def build_index(self, embeddings: np.ndarray) -> None:
        if self.is_trained:
            self.clean()
        self.index.db = embeddings
        ids = list(np.arange(len(embeddings)))
        ids = [str(i) for i in ids]
        self.index = self.index.build(docids=ids)
        self.index.set_num_threads(self.threads)
        self.serialize()
        return

    def _prepare_index(self):
        if self.distance_function == "IP":
            distance_measure = "dot_product"
        else:
            distance_measure = "squared_l2"
        builder = (
            self.scann.scann_ops_pybind.builder(
                None,
                self.num_neighbors,
                distance_measure=distance_measure,
            )
            .tree(
                num_leaves=self.num_leaves,
                num_leaves_to_search=self.num_leaves_to_search,
                training_sample_size=self.index_train_num,
            )
            .score_ah(
                dimensions_per_block=2,
                anisotropic_quantization_threshold=self.anisotropic_quantization_threshold,
            )
            .reorder(200)
        )
        builder.set_n_training_threads(self.threads)
        return builder

    def _add_embeddings_batch(self, embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype("float32")
        assert self.is_trained, "Index should be trained first"
        ids = list(range(len(self), len(self) + len(embeddings)))
        ids = [str(i) for i in ids]
        self.index.upsert(docids=ids, database=embeddings, batch_size=self.batch_size)
        return

    def _search_batch(
        self,
        query: np.ndarray,
        top_k: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = query.astype("float32")
        indices, scores = self.index.search_batched(query, top_k, **search_kwargs)
        indices = np.array([[int(i) for i in idx] for idx in indices])
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

    def clean(self):
        if not self.is_trained:
            return
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        self.index = self._prepare_index()
        return

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
