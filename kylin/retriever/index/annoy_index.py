import logging
import os
import shutil
from dataclasses import dataclass

import numpy as np
from tables import EArray

from kylin.utils import Choices, SimpleProgressLogger

from .index_base import DENSE_INDEX, DenseIndex, DenseIndexConfigBase

logger = logging.getLogger(__name__)


@dataclass
class AnnoyIndexConfig(DenseIndexConfigBase):
    distance_function: Choices(["IP", "L2", "COSINE", "HAMMING", "MANHATTAN"]) = "IP"  # type: ignore
    n_trees: int = 100
    n_jobs: int = -1
    search_k: int = 1000


@DENSE_INDEX("annoy", config_class=AnnoyIndexConfig)
class AnnoyIndex(DenseIndex):
    def __init__(
        self, index_path: str, embedding_size: int, cfg: AnnoyIndexConfig
    ) -> None:
        super().__init__(index_path, embedding_size, cfg)
        # check annoy
        try:
            from annoy import AnnoyIndex as AnnIndex

            match cfg.distance_function:
                case "IP":
                    self.index = AnnIndex(self.embedding_size, "dot")
                case "L2":
                    self.index = AnnIndex(self.embedding_size, "euclidean")
                case "COSINE":
                    self.index = AnnIndex(self.embedding_size, "angular")
                case "HAMMING":
                    self.index = AnnIndex(self.embedding_size, "hamming")
                case "MANHATTAN":
                    self.index = AnnIndex(self.embedding_size, "manhattan")
        except:
            raise ImportError("Please install annoy by running `pip install annoy`")

        # set annoy params
        self.n_jobs = cfg.n_jobs
        self.n_trees = cfg.n_trees
        self.search_k = cfg.search_k

        # prepare index
        if os.path.exists(self.index_path):
            self.deserialize()
        else:
            if not os.path.exists(os.path.dirname(self.index_path)):
                os.makedirs(os.path.dirname(self.index_path))
            self.index.on_disk_build(self.index_path)
        return

    def build_index(self, embeddings: np.ndarray | EArray) -> None:
        assert not self.is_trained, "Index is already trained"
        p_logger = SimpleProgressLogger(
            logger, total=len(embeddings), interval=self.log_interval
        )
        for idx, embed in enumerate(embeddings):
            self.index.add_item(idx, embed)
            p_logger.update(step=1, desc="Adding embeddings")
        logger.info("Building index")
        self.index.build(self.n_trees, self.n_jobs)
        return

    def _add_embeddings_batch(self, embeddings: np.ndarray) -> None:
        raise NotImplementedError(
            "Annoy does not support adding embeddings. Please retrain the index."
        )

    def _search_batch(
        self,
        query: np.ndarray,
        top_k: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = query.astype("float32")
        indices = []
        scores = []
        search_k = search_kwargs.get("search_k", self.search_k)
        for q in query:
            idx, dis = self.index.get_nns_by_vector(
                q,
                top_k,
                search_k=search_k,
                include_distances=True,
            )
            indices.append(idx)
            scores.append(dis)
        indices = np.array(indices)
        scores = np.array(scores)
        return indices, scores

    def serialize(self) -> None:
        logger.info(f"Serializing index to {self.index_path}")
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
        self.index.save(self.index_path)
        return

    def deserialize(self) -> None:
        logger.info(f"Loading index from {self.index_path}")
        self.index.load(self.index_path)
        assert self.index.f == self.embedding_size, "Index dimension mismatch"
        return

    def clean(self):
        self.index.unload()
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        return

    @property
    def is_trained(self):
        return self.index.get_n_items() > 0

    def __len__(self) -> int:
        return self.index.get_n_items()
