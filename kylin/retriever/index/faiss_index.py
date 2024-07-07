import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .index_base import DenseIndex, DenseIndexConfig

logger = logging.getLogger(__name__)


@dataclass
class FaissIndexConfig(DenseIndexConfig):
    n_subquantizers: int = 0
    n_bits: int = 0
    n_list: int = 1000
    n_probe: int = 32
    factory_str: Optional[str] = None
    device_id: list[int] = field(default_factory=list)


class FaissIndex(DenseIndex):
    def __init__(self, index_path: str, cfg: FaissIndexConfig) -> None:
        super().__init__(index_path, cfg)
        # check faiss
        try:
            import faiss

            self.faiss = faiss
        except:
            raise ImportError(
                "Please install faiss by running `conda install faiss-gpu`"
            )

        # preapre basic args
        self.n_subquantizers = cfg.n_subquantizers
        self.n_bits = cfg.n_bits
        self.n_list = cfg.n_list
        self.nprobe = cfg.n_probe
        self.factory_str = cfg.factory_str

        # prepare GPU resource
        self.device_id = cfg.device_id
        if self.device_id != [-1]:
            self.gpu_res = self.faiss.StandardGpuResources()
            self.gpu_option = self.faiss.GpuClonerOptions()
            self.gpu_option.useFloat16 = True
        else:
            self.gpu_res = None
            self.gpu_option = None

        # prepare index
        if os.path.exists(self.index_path):
            self.deserialize()
        else:
            index = self._prepare_index()
            self._set_index(index)
        return

    def build_index(self, embeddings: np.ndarray, ids: np.ndarray | list[int] = None):
        self.train_index(embeddings=embeddings)
        self.add_embeddings(embeddings=embeddings, ids=ids)
        self.serialize()
        return

    def _prepare_index(self):
        # prepare basic index
        if self.distance_function == "IP":
            basic_index = self.faiss.IndexFlatIP(self.embedding_size)
            basic_metric = self.faiss.METRIC_INNER_PRODUCT
        else:
            assert self.distance_function == "L2"
            basic_index = self.faiss.IndexFlatL2(self.embedding_size)
            basic_metric = self.faiss.METRIC_L2

        # prepare optimized index
        if self.factory_str is not None:
            basic_index = self.faiss.index_factory(
                self.embedding_size,
                self.factory_str,
                basic_metric,
            )
        elif (self.n_list > 0) and (self.n_bits > 0):  # IVFPQ
            basic_index = self.faiss.IndexIVFPQ(
                basic_index,
                self.embedding_size,
                self.n_list,
                self.n_subquantizers,
                self.n_bits,
            )
            basic_index.nprobe = self.n_probe
        elif self.n_bits > 0:  # PQ
            basic_index = self.faiss.IndexPQ(
                self.embedding_size,
                self.n_subquantizers,
                self.n_bits,
            )
        elif self.n_list > 0:  # IVF
            basic_index = self.faiss.IndexIVFFlat(
                basic_index,
                self.embedding_size,
                self.n_list,
                basic_metric,
            )
            basic_index.nprobe = self.n_probe
        return basic_index

    def train_index(self, embeddings: np.ndarray) -> None:
        logger.info("Training index")
        embeddings = embeddings.astype("float32")
        train_num = min(self.index_train_num, embeddings.shape[0])
        selected_indices = np.random.choice(
            embeddings.shape[0],
            train_num,
            replace=False,
        )
        self.index.train(embeddings[selected_indices])
        return

    def _add_embeddings_batch(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray | list[int],
    ) -> None:
        embeddings = embeddings.astype("float32")
        assert self.is_trained, "Index should be trained first"
        assert len(embeddings) == len(ids)
        self.index.add_with_ids(len(embeddings), embeddings, ids)
        return

    def _search_batch(
        self,
        query_vectors: np.ndarray,
        top_docs: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = query_vectors.astype("float32")
        return self.index.search(query_vectors, top_docs, **search_kwargs)

    def serialize(self) -> None:
        logger.info(f"Serializing index to {self.index_path}")
        if self.device_id >= 0:
            cpu_index = self.faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        self.faiss.write_index(cpu_index, self.index_path)
        return

    def deserialize(self) -> None:
        logger.info(f"Loading index from {self.index_path}")
        if os.path.getsize(self.index_path) / (1024**3) > 10:
            logger.info("Index file is too large. Loading on CPU with memory map.")
            self.device_id = -1
            self.gpu_res = None
            self.gpu_option = None
            cpu_index = self.faiss.read_index(self.index_path, self.faiss.IO_FLAG_MMAP)
        else:
            cpu_index = self.faiss.read_index(self.index_path)
        cpu_index.nprobe = self.nprobe
        self._set_index(cpu_index)
        return

    def clear(self):
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        self.index = self._prepare_index()
        self._set_index(self.index)
        return

    @property
    def is_trained(self):
        return self.index.is_trained

    @property
    def embedding_size(self):
        return self.index.d

    def __len__(self):
        return self.index.ntotal

    # TODO: implement _set_index method
    def _set_index(self, index) -> None:
        raise NotImplementedError
        if self.gpu_res is not None:
            faiss.index_cpu_to_all_gpus(
                index,
            )
            self.index = faiss.index_cpu_to_gpu(
                self.gpu_res,
                self.device_id,
                index,
                self.gpu_option,
            )
        else:
            self.index = index
        return
