import math
import os
import shutil
from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable

import numpy as np
from omegaconf import OmegaConf

from flexrag.utils import LOGGER_MANAGER, SimpleProgressLogger

from .index_base import RETRIEVER_INDEX, DenseIndexBase, DenseIndexBaseConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.index.annoy")


@dataclass
class AnnoyIndexConfig(DenseIndexBaseConfig):
    """The configuration for the `AnnoyIndex`.

    :param n_trees: The number of trees to build the index. Defaults to -1.
        -1 means auto.
        The number of trees is set to max(1, log(n_items) // 10) * sqrt(embedding_size) * 10.
    :type n_trees: int
    :param n_jobs: The number of jobs to build the index. Defaults to -1.
        -1 means all available CPU cores.
    :type n_jobs: int
    :param search_k: The number of neighbors to search. Defaults to -1.
        -1 means auto.
        The number of neighbors to search is set to max(top_k, 100) * n_trees.
    :type search_k: int
    :param on_disk_build: Whether to build the index on disk. Defaults to False.
    :type on_disk_build: bool
    """

    n_trees: int = -1  # -1 means auto
    n_jobs: int = -1  # -1 means auto
    n_neighbors: int = -1  # -1 means auto
    on_disk_build: bool = False


@RETRIEVER_INDEX("annoy", config_class=AnnoyIndexConfig)
class AnnoyIndex(DenseIndexBase):
    """AnnoyIndex is a wrapper for the `annoy <https://github.com/spotify/annoy>`_ library.

    AnnoyIndex supports building index on disk, which is useful when the memory is limited.
    However, building index on disk is slower than building index in memory.
    """

    def __init__(self, cfg: AnnoyIndexConfig) -> None:
        super().__init__(cfg)
        self.cfg = AnnoyIndexConfig.extract(cfg)
        # check annoy
        try:
            from annoy import AnnoyIndex as AnnIndex

            self.ann = AnnIndex
        except:
            raise ImportError("Please install annoy by running `pip install annoy`")

        # set annoy params
        if self.cfg.on_disk_build:
            assert (
                self.cfg.index_path is not None
            ), "index_path is required for on disk build."
        match self.cfg.distance_function:
            case "IP":
                self.index = partial(self.ann, metric="dot")
            case "L2":
                self.index = partial(self.ann, metric="euclidean")
            case _:
                raise ValueError(
                    f"Unsupported distance function: {self.cfg.distance_function}"
                )

        # load the index if index_path is provided
        if self.cfg.index_path is not None:
            if os.path.exists(self.cfg.index_path):
                logger.info(f"Loading index from {self.cfg.index_path}")
                try:
                    meta_path = os.path.join(self.cfg.index_path, "emb.size")
                    emb_size = int(
                        open(meta_path, "r", encoding="utf-8").read().strip()
                    )
                    self.index = self.index(emb_size)
                    self.index.load(os.path.join(self.cfg.index_path, "index.annoy"))
                except:
                    raise FileNotFoundError(
                        f"Unable to load index from {self.cfg.index_path}"
                    )
        return

    def build_index(self, data: Iterable[Any]) -> None:
        self.clear()
        embeddings = self.encode_data_batch(data, is_query=False)
        # prepare index
        self.index = self.index(embeddings.shape[1])
        if self.cfg.on_disk_build:
            self.index.on_disk_build(os.path.join(self.cfg.index_path, "index.annoy"))

        # add embeddings
        p_logger = SimpleProgressLogger(logger, len(embeddings), self.cfg.log_interval)
        for n, embed in enumerate(embeddings):
            self.index.add_item(n, embed)
            p_logger.update(step=1, desc="Adding embeddings")

        # build index
        logger.info("Building index")
        if self.cfg.n_trees == -1:
            n_trees = (
                max(1, math.floor(math.log(embeddings.shape[0]) // 10))
                * math.floor(math.sqrt(embeddings.shape[1]))
                * 10
            )
        else:
            n_trees = self.cfg.n_trees
        self.index.build(n_trees, self.cfg.n_jobs)

        # serialize index
        if self.cfg.index_path is not None:
            self.save_to_local()

        # clear temporary file
        if isinstance(embeddings, np.memmap):
            os.remove(embeddings.filename)
            del embeddings
        return

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        raise NotImplementedError(
            "AnnoyIndex does not support adding embeddings. Please retrain the index."
        )

    def search(
        self,
        query: list[Any],
        top_k: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        # encode query
        query_vector = self.encode_data(query, is_query=True)

        # prepare search params
        n_neighbors = search_kwargs.get("n_neighbors", self.cfg.n_neighbors)
        if n_neighbors == -1:
            n_neighbors = max(top_k, 100) * self.cfg.n_trees

        indices = []
        scores = []
        for q in query_vector:
            idx, dis = self.index.get_nns_by_vector(
                q,
                top_k,
                search_k=n_neighbors,
                include_distances=True,
            )
            indices.append(idx)
            scores.append(dis)
        indices = np.array(indices)
        scores = np.array(scores)
        return indices, scores

    def save_to_local(self, index_path: str = None) -> None:
        if index_path is not None:
            self.cfg.index_path = index_path
        assert self.cfg.index_path is not None, "`index_path` is not set."
        if not os.path.exists(self.cfg.index_path):
            os.makedirs(self.cfg.index_path)
        logger.info(f"Serializing index to {self.cfg.index_path}")

        # save configurations
        cfg_path = os.path.join(self.cfg.index_path, "config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            OmegaConf.save(self.cfg, f)
        id_path = os.path.join(self.cfg.index_path, "cls.id")
        with open(id_path, "w", encoding="utf-8") as f:
            f.write(self.__class__.__name__)
        additional_cfg_path = os.path.join(self.cfg.index_path, "emb.size")
        with open(additional_cfg_path, "w", encoding="utf-8") as f:
            f.write(str(self.embedding_size))

        # save index
        if self.cfg.on_disk_build:
            return
        index_path = os.path.join(self.cfg.index_path, "index.annoy")
        self.index.save(index_path)
        return

    def clear(self):
        if not isinstance(self.index, partial):
            self.index.unload()
        if self.cfg.index_path is not None:
            if os.path.exists(self.cfg.index_path):
                shutil.rmtree(self.cfg.index_path)
        return

    @property
    def embedding_size(self) -> int:
        if not isinstance(self.index, partial):
            return self.index.f
        if self.passage_encoder is not None:
            return self.passage_encoder.embedding_size
        if self.query_encoder is not None:
            return self.query_encoder.embedding_size
        raise ValueError("Index is not initialized.")

    @property
    def is_addable(self) -> bool:
        return False

    @property
    def n_trees(self) -> int:
        if not hasattr(self, "index"):
            return self.cfg.n_trees
        if isinstance(self.index, partial):
            return self.cfg.n_trees
        if self.index.get_n_items() <= 0:
            return self.cfg.n_trees
        if self.index.get_n_trees() == 0:
            return self.cfg.n_trees
        return self.index.get_n_trees()

    def __len__(self) -> int:
        if isinstance(self.index, partial):
            return 0
        return self.index.get_n_items()
