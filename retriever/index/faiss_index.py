import logging
import os
from argparse import ArgumentParser

import faiss
import numpy as np

from .index import DenseIndex


logger = logging.getLogger(__name__)


class FaissIndex(DenseIndex):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--n_subquantizers",
            type=int,
            default=0,
            help="Number of subquantizers for IVFPQ index",
        )
        parser.add_argument(
            "--n_bits",
            type=int,
            default=0,
            help="Number of bits for PQ index",
        )
        parser.add_argument(
            "--n_list",
            type=int,
            default=1000,
            help="Number of clusters for IVF index",
        )
        parser.add_argument(
            "--n_probe",
            type=int,
            default=32,
            help="Number of clusters to explore at search time",
        )
        parser.add_argument(
            "--factory_str",
            type=str,
            default=None,
            help="String to pass to index factory",
        )
        return parser

    def __init__(
        self,
        index_path: str,
        embedding_size: int = 768,
        n_subquantizers: int = 0,
        n_bits: int = 8,
        n_list: int = 1000,
        n_probe: int = 36,
        distance_func: str = "IP",
        factory_str: str = None,
        device_id: int = -1,
        index_train_num: int = 1000000,
        log_interval: int = 100,
        **kwargs,
    ) -> None:
        # preapre basic args
        self.index_train_num = index_train_num
        self.log_interval = log_interval
        self.index_path = os.path.join(index_path, "index.faiss")
        self.nprobe = n_probe

        # prepare GPU resource
        self.device_id = device_id
        if self.device_id >= 0:
            self.res = faiss.StandardGpuResources()
            self.gpu_option = faiss.GpuClonerOptions()
            self.gpu_option.useFloat16 = True
        else:
            self.res = None
            self.gpu_option = None

        # prepare index
        if self.index_path is not None and os.path.exists(self.index_path):
            self.deserialize()
        else:
            index = self.prepare_index(
                embedding_size=embedding_size,
                n_subquantizers=n_subquantizers,
                n_bits=n_bits,
                n_list=n_list,
                n_probe=n_probe,
                factory_str=factory_str,
                distance_func=distance_func,
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
    ) -> faiss.Index:
        # prepare basic index
        if distance_func == "IP":
            basic_index = faiss.IndexFlatIP(embedding_size)
            basic_metric = faiss.METRIC_INNER_PRODUCT
        else:
            assert distance_func == "L2"
            basic_index = faiss.IndexFlatL2(embedding_size)
            basic_metric = faiss.METRIC_L2

        # prepare optimized index
        if factory_str is not None:
            basic_index = faiss.index_factory(
                embedding_size,
                factory_str,
                basic_metric,
            )
        elif (n_list > 0) and (n_bits > 0):  # IVFPQ
            basic_index = faiss.IndexIVFPQ(
                basic_index,
                embedding_size,
                n_list,
                n_subquantizers,
                n_bits,
            )
            basic_index.nprobe = n_probe
        elif n_bits > 0:  # PQ
            basic_index = faiss.IndexPQ(
                embedding_size,
                n_subquantizers,
                n_bits,
            )
        elif n_list > 0:  # IVF
            basic_index = faiss.IndexIVFFlat(
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
        batch_size: int = 512,
        ids: np.ndarray | list[int] = None,
    ):
        if ids is not None:
            assert len(ids) == len(embeddings)
        for idx in range(0, len(embeddings), batch_size):
            if (idx // batch_size) % self.log_interval == 0:
                logger.info(f"Adding {idx} / {len(embeddings)} embeddings.")
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
        assert self.index.is_trained, "Index should be trained first"
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
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        faiss.write_index(cpu_index, self.index_path)
        return

    def deserialize(self) -> None:
        logger.info(f"Loading index from {self.index_path}")
        if os.path.getsize(self.index_path) / (1024**3) > 10:
            logger.info("Index file is too large. Loading on CPU with memory map.")
            self.device_id = -1
            self.res = None
            self.gpu_option = None
            cpu_index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
        else:
            cpu_index = faiss.read_index(self.index_path)
        cpu_index.nprobe = self.nprobe
        self._set_index(cpu_index)
        return

    @property
    def is_trained(self):
        return self.index.is_trained

    def __len__(self):
        return self.index.ntotal
    
    def _set_index(self, index: faiss.Index) -> None:
        if self.device_id >= 0:
            self.index = faiss.index_cpu_to_gpu(
                self.res,
                self.device_id,
                index,
                self.gpu_option,
            )
        else:
            self.index = index
        return
