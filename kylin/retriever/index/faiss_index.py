import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from tables import EArray

from kylin.utils import Choices

from .index_base import DenseIndex, DenseIndexConfigBase, DENSE_INDEX

logger = logging.getLogger(__name__)


@dataclass
class FaissIndexConfig(DenseIndexConfigBase):
    index_type: Choices(["FLAT", "IVF", "PQ", "IVFPQ"]) = "FLAT"  # type: ignore
    n_subquantizers: int = 8
    n_bits: int = 8
    n_list: int = 1000
    n_probe: int = 32
    factory_str: Optional[str] = None
    device_id: list[int] = field(default_factory=list)


@DENSE_INDEX("faiss", config_class=FaissIndexConfig)
class FaissIndex(DenseIndex):
    def __init__(
        self, index_path: str, embedding_size: int, cfg: FaissIndexConfig
    ) -> None:
        super().__init__(index_path, embedding_size, cfg)
        # check faiss
        try:
            import faiss

            self.faiss = faiss
        except:
            raise ImportError(
                "Please install faiss by running `conda install -c pytorch -c nvidia faiss-gpu`"
            )

        # preapre basic args
        self.n_probe = cfg.n_probe
        self.device_id = cfg.device_id

        # prepare index
        if os.path.exists(self.index_path):
            self.index = self.deserialize()
        else:
            self.index = self._prepare_index(
                index_type=cfg.index_type,
                distance_function=cfg.distance_function,
                embedding_size=self.embedding_size,
                n_list=cfg.n_list,
                n_probe=cfg.n_probe,
                n_subquantizers=cfg.n_subquantizers,
                n_bits=cfg.n_bits,
                factory_str=cfg.factory_str,
            )
        return

    def build_index(self, embeddings: np.ndarray | EArray) -> None:
        self.clean()
        if isinstance(embeddings, EArray):
            embeddings = embeddings.read()
        self.train_index(embeddings=embeddings)
        self.add_embeddings(embeddings=embeddings)
        return

    def _prepare_index(
        self,
        index_type: str,
        distance_function: str,
        embedding_size: int,
        n_list: int,  # the number of cells
        n_probe: int,  # the number of cells to visit
        n_subquantizers: int,  # the number of subquantizers
        n_bits: int,  # the number of bits per subquantizer
        factory_str: Optional[str] = None,
    ):
        # prepare distance function
        match distance_function:
            case "IP":
                basic_index = self.faiss.IndexFlatIP(embedding_size)
                basic_metric = self.faiss.METRIC_INNER_PRODUCT
            case "L2":
                basic_index = self.faiss.IndexFlatL2(embedding_size)
                basic_metric = self.faiss.METRIC_L2
            case _:
                raise ValueError(f"Unknown distance function: {distance_function}")

        if factory_str is not None:
            # using string factory to build the index
            index = self.faiss.index_factory(
                embedding_size,
                factory_str,
                basic_metric,
            )
        else:
            # prepare optimized index
            match index_type:
                case "FLAT":
                    index = basic_index
                case "IVF":
                    index = self.faiss.IndexIVFFlat(
                        basic_index,
                        embedding_size,
                        n_list,
                        basic_metric,
                    )
                    index.nprobe = n_probe
                case "PQ":
                    index = self.faiss.IndexPQ(
                        embedding_size,
                        n_subquantizers,
                        n_bits,
                    )
                case "IVFPQ":
                    index = self.faiss.IndexIVFPQ(
                        basic_index,
                        embedding_size,
                        n_list,
                        n_subquantizers,
                        n_bits,
                    )
                    index.nprobe = self.n_probe
                case _:
                    raise ValueError(f"Unknown index type: {index_type}")

        # post process
        index = self._set_index(index)
        return index

    def train_index(self, embeddings: np.ndarray) -> None:
        if self.is_flat:
            logger.info("Index is flat, no need to train")
            return
        logger.info("Training index")
        embeddings = embeddings.astype("float32")
        if (self.index_train_num >= embeddings.shape[0]) or (
            self.index_train_num == -1
        ):
            self.index.train(embeddings)
        else:
            selected_indices = np.random.choice(
                embeddings.shape[0],
                self.index_train_num,
                replace=False,
            )
            self.index.train(embeddings[selected_indices])
        return

    def _add_embeddings_batch(self, embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype("float32")
        assert self.is_trained, "Index should be trained first"
        self.index.add(embeddings)  # debug
        return

    def _search_batch(
        self,
        query_vectors: np.ndarray,
        top_docs: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = query_vectors.astype("float32")
        scores, indices = self.index.search(query_vectors, top_docs, **search_kwargs)
        return indices, scores

    def serialize(self) -> None:
        logger.info(f"Serializing index to {self.index_path}")
        if len(self.device_id) >= 0:
            cpu_index = self.faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        self.faiss.write_index(cpu_index, self.index_path)
        return

    def deserialize(self):
        logger.info(f"Loading index from {self.index_path}.")
        if (os.path.getsize(self.index_path) / (1024**3) > 10) and (
            len(self.device_id) == 0
        ):
            logger.info("Index file is too large. Loading on CPU with memory map.")
            cpu_index = self.faiss.read_index(self.index_path, self.faiss.IO_FLAG_MMAP)
        else:
            cpu_index = self.faiss.read_index(self.index_path)
        if hasattr(cpu_index, "nprobe"):
            cpu_index.nprobe = self.n_probe
        index = self._set_index(cpu_index)
        assert index.d == self.embedding_size, "Index dimension mismatch"
        return index

    def clean(self):
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        self.index.reset()
        return

    @property
    def is_trained(self) -> bool:
        if isinstance(self.index, self.faiss.IndexReplicas):
            trained = True
            for i in range(self.index.count()):
                sub_index = self.faiss.downcast_index(self.index.at(i))
                if not sub_index.is_trained:
                    trained = False
            return trained
        return self.index.is_trained

    @property
    def is_flat(self) -> bool:
        if isinstance(
            self.index,
            (self.faiss.IndexFlat, self.faiss.GpuIndexFlat),
        ):
            return True
        if isinstance(self.index, self.faiss.IndexReplicas):
            all_flat = True
            for i in range(self.index.count()):
                sub_index = self.faiss.downcast_index(self.index.at(i))
                if not isinstance(
                    sub_index,
                    (self.faiss.IndexFlat, self.faiss.GpuIndexFlat),
                ):
                    all_flat = False
            if all_flat:
                return True
        return False

    def __len__(self):
        return self.index.ntotal

    def _set_index(self, index):
        if len(self.device_id) > 0:
            logger.info("Accelerating index with GPU.")
            option = self.faiss.GpuMultipleClonerOptions()
            option.useFloat16 = True
            option.shard = True
            if isinstance(index, self.faiss.IndexIVFFlat):
                option.common_ivf_quantizer = True
            index = self.faiss.index_cpu_to_gpus_list(
                index,
                co=option,
                gpus=self.device_id,
                ngpu=len(self.device_id),
            )
        return index
