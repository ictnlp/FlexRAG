import os
import re
import shutil
from dataclasses import dataclass

import numpy as np

from flexrag.utils import LOGGER_MANAGER

from .index_base import DENSE_INDEX, DenseIndexBase, DenseIndexBaseConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.index.scann")


@dataclass
class ScaNNIndexConfig(DenseIndexBaseConfig):
    num_leaves: int = 2000
    num_leaves_to_search: int = 500
    num_neighbors: int = 10
    anisotropic_quantization_threshold: float = 0.2
    dimensions_per_block: int = 2
    threads: int = 0


@DENSE_INDEX("scann", config_class=ScaNNIndexConfig)
class ScaNNIndex(DenseIndexBase):
    def __init__(self, cfg: ScaNNIndexConfig, index_path: str) -> None:
        super().__init__(cfg, index_path)
        # check scann
        try:
            import scann

            self.scann = scann
        except:
            raise ImportError("Please install scann by running `pip install scann`")

        # set basic args
        self.cfg = cfg

        # prepare index
        if os.path.exists(self.index_path):
            self.deserialize()
        else:
            self.index = None
        return

    def build_index(self, embeddings: np.ndarray) -> None:
        # prepare arguments
        if self.is_trained:
            self.clean()
        if self.cfg.distance_function == "IP":
            distance_measure = "dot_product"
        else:
            distance_measure = "squared_l2"
        train_num = (
            len(embeddings)
            if self.cfg.index_train_num <= 0
            else self.cfg.index_train_num
        )
        # prepare builder
        builder = (
            self.scann.scann_ops_pybind.builder(
                embeddings,
                self.cfg.num_neighbors,
                distance_measure=distance_measure,
            )
            .tree(
                num_leaves=self.cfg.num_leaves,
                num_leaves_to_search=self.cfg.num_leaves_to_search,
                training_sample_size=train_num,
            )
            .score_ah(
                dimensions_per_block=self.cfg.dimensions_per_block,
                anisotropic_quantization_threshold=self.cfg.anisotropic_quantization_threshold,
            )
            .reorder(200)
        )
        builder.set_n_training_threads(self.cfg.threads)
        ids = list(np.arange(len(embeddings)))
        ids = [str(i) for i in ids]
        # build index
        self.index = builder.build(docids=ids)
        self.index.set_num_threads(self.cfg.threads)
        self.serialize()
        return

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
        assert self.is_trained, "Index should be trained first."
        logger.info(f"Serializing index to {self.index_path}")
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
        self.index.serialize(self.index_path)
        return

    def deserialize(self) -> None:
        logger.info(f"Loading index from {self.index_path}.")
        self.index = self.scann.scann_ops_pybind.load_searcher(self.index_path)
        return

    def clean(self):
        if not self.is_trained:
            return
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        self.index = None
        return

    @property
    def embedding_size(self) -> int:
        if self.index is None:
            raise RuntimeError("Index is not built yet.")
        return int(re.search("input_dim: [0-9]+", self.index.config()).group()[11:])

    @property
    def is_trained(self) -> bool:
        if self.index is None:
            return False
        return not isinstance(self.index, self.scann.ScannBuilder)

    def __len__(self) -> int:
        if self.index is None:
            return 0
        if isinstance(self.index, self.scann.ScannBuilder):
            return 0
        return self.index.size()
