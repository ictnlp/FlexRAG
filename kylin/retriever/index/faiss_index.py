import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from kylin.utils import SimpleProgressLogger

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
    def __init__(self, cfg: FaissIndexConfig) -> None:
        super().__init__(cfg)
        # check faiss
        try:
            import faiss

            self.faiss = faiss
        except:
            raise ImportError("Please install faiss by running `conda install faiss-gpu`")

        # preapre basic args
        self.index_train_num = cfg.index_train_num
        self.log_interval = cfg.log_interval
        self.index_path = cfg.index_path
        self.nprobe = cfg.n_probe

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
        if self.index_path is not None and os.path.exists(self.index_path):
            self.deserialize()
        else:
            index = self.prepare_index(
                embedding_size=cfg.embedding_size,
                n_subquantizers=cfg.n_subquantizers,
                n_bits=cfg.n_bits,
                n_list=cfg.n_list,
                n_probe=cfg.n_probe,
                factory_str=cfg.factory_str,
                distance_func=self.distance_function,
            )
            self._set_index(index)
        return

    def prepare_index(
        self,
        embedding_size: int,
        n_subquantizers: int,
        n_bits: int,
        n_list: int,
        n_probe: int,
        factory_str: str,
        distance_func: str = "IP",
    ):
        # prepare basic index
        if distance_func == "IP":
            basic_index = self.faiss.IndexFlatIP(embedding_size)
            basic_metric = self.faiss.METRIC_INNER_PRODUCT
        else:
            assert distance_func == "L2"
            basic_index = self.faiss.IndexFlatL2(embedding_size)
            basic_metric = self.faiss.METRIC_L2

        # prepare optimized index
        if factory_str is not None:
            basic_index = self.faiss.index_factory(
                embedding_size,
                factory_str,
                basic_metric,
            )
        elif (n_list > 0) and (n_bits > 0):  # IVFPQ
            basic_index = self.faiss.IndexIVFPQ(
                basic_index,
                embedding_size,
                n_list,
                n_subquantizers,
                n_bits,
            )
            basic_index.nprobe = n_probe
        elif n_bits > 0:  # PQ
            basic_index = self.faiss.IndexPQ(
                embedding_size,
                n_subquantizers,
                n_bits,
            )
        elif n_list > 0:  # IVF
            basic_index = self.faiss.IndexIVFFlat(
                basic_index,
                embedding_size,
                n_list,
                basic_metric,
            )
            basic_index.nprobe = n_probe
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

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray | list[int] = None,
        batch_size: int = 512,
    ):
        if ids is not None:
            assert len(ids) == len(embeddings)

        p_logger = SimpleProgressLogger(
            logger, total=embeddings.shape[0], interval=self.log_interval
        )
        for idx in range(0, len(embeddings), batch_size):
            p_logger.update(step=batch_size, desc="Adding embeddings")
            embeds_to_add = embeddings[idx : idx + batch_size]
            if ids is not None:
                ids_to_add = ids[idx : idx + batch_size]
                self.add_embeddings_batch(embeds_to_add, ids_to_add)
            else:
                self.add_embeddings_batch(embeds_to_add)
        return

    def add_embeddings_batch(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray | list[int] = None,
    ) -> None:
        embeddings = embeddings.astype("float32")
        assert self.is_trained, "Index should be trained first"
        if ids is not None:
            assert len(embeddings) == len(ids)
            self.index.add_with_ids(len(embeddings), embeddings, ids)
        else:
            self.index.add(embeddings)
        return

    def search(
        self,
        query_vector: np.ndarray,
        top_docs: int,
        batch_size: int = 512,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = []
        indices = []
        for idx in range(0, len(query_vector), batch_size):
            if (idx // batch_size) % self.log_interval == 0:
                logger.info(f"Searching {idx} / {len(query_vector)} queries.")
            query = query_vector[idx : idx + batch_size]
            r = self.search_batch(query, top_docs, **search_kwargs)
            scores.append(r[0])
            indices.append(r[1])
        scores = np.concatenate(scores, axis=0)
        indices = np.concatenate(indices, axis=0)
        return scores, indices

    def search_batch(
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

    # TODO: implement clear method
    def clear(self):
        raise NotImplementedError

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
