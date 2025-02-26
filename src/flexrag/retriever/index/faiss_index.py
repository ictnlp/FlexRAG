import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from flexrag.utils import Choices, LOGGER_MANAGER

from .index_base import DenseIndexBase, DenseIndexBaseConfig, DENSE_INDEX

logger = LOGGER_MANAGER.get_logger("flexrag.retriever.index.faiss")


@dataclass
class FaissIndexConfig(DenseIndexBaseConfig):
    """The configuration for the `FaissIndex`.

    :param index_type: The type of the index. Defaults to "auto".
        available choices are "FLAT", "IVF", "PQ", "IVFPQ", and "auto".
        If set to "auto", the index will be set to "IVF{n_list},PQ{embedding_size//2}x4fs".
    :type index_type: str
    :param n_subquantizers: The number of subquantizers. Defaults to 8.
    :type n_subquantizers: int
    :param n_bits: The number of bits per subquantizer. Defaults to 8.
    :type n_bits: int
    :param n_list: The number of cells. Defaults to 1000.
    :type n_list: int
    :param factory_str: The factory string to build the index. Defaults to None.
        If set, the `index_type` will be ignored.
    :type factory_str: Optional[str]
    :param n_probe: The number of probes. Defaults to 32.
    :type n_probe: int
    :param device_id: The device id to use. Defaults to [].
        [] means CPU. If set, the index will be accelerated with GPU.
    :type device_id: list[int]
    :param k_factor: The k factor for search. Defaults to 10.
    :type k_factor: int
    :param polysemous_ht: The polysemous hash table. Defaults to 0.
    :type polysemous_ht: int
    :param efSearch: The efSearch for HNSW. Defaults to 100.
    :type efSearch: int
    """

    index_type: Choices(["FLAT", "IVF", "PQ", "IVFPQ", "auto"]) = "auto"  # type: ignore
    n_subquantizers: int = 8
    n_bits: int = 8
    n_list: int = 1000
    factory_str: Optional[str] = None
    # Inference Arguments
    n_probe: int = 32
    device_id: list[int] = field(default_factory=list)
    k_factor: int = 10
    polysemous_ht: int = 0
    efSearch: int = 100


@DENSE_INDEX("faiss", config_class=FaissIndexConfig)
class FaissIndex(DenseIndexBase):
    """FaissIndex is a wrapper for the `faiss <https://github.com/facebookresearch/faiss>`_ library.

    FaissIndex provides a flexible and efficient way to build and search indexes with embeddings.
    """

    def __init__(self, cfg: FaissIndexConfig, index_path: str) -> None:
        super().__init__(cfg, index_path)
        # check faiss
        try:
            import faiss

            self.faiss = faiss
        except:
            raise ImportError(
                "Please install faiss by running "
                "`conda install -c pytorch faiss-cpu=1.9.0` "
                "or `conda install -c pytorch -c nvidia faiss-gpu`"
            )

        # preapre inference args
        self.n_probe = cfg.n_probe
        self.device_id = (
            cfg.device_id if hasattr(self.faiss, "GpuMultipleClonerOptions") else []
        )
        self.k_factor = cfg.k_factor
        self.polysemous_ht = cfg.polysemous_ht
        self.efSearch = cfg.efSearch

        # prepare index args
        self.index_type = cfg.index_type
        self.distance_function = cfg.distance_function
        self.n_list = cfg.n_list
        self.n_subquantizers = cfg.n_subquantizers
        self.n_bits = cfg.n_bits
        self.factory_str = cfg.factory_str

        # load the index if exists
        self.index = None
        if self.index_path is not None:
            if os.path.exists(self.index_path):
                self.deserialize()
        return

    def build_index(self, embeddings: np.ndarray) -> None:
        self.clean()
        self.index = self._prepare_index(
            index_type=self.index_type,
            distance_function=self.distance_function,
            embedding_size=embeddings.shape[1],
            embedding_length=embeddings.shape[0],
            n_list=self.n_list,
            n_subquantizers=self.n_subquantizers,
            n_bits=self.n_bits,
            factory_str=self.factory_str,
        )
        self.train_index(embeddings=embeddings)
        self.add_embeddings(embeddings=embeddings)
        return

    def _prepare_index(
        self,
        index_type: str,
        distance_function: str,
        embedding_size: int,  # the dimension of the embeddings
        embedding_length: int,  # the number of the embeddings
        n_list: int,  # the number of cells
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

        if index_type == "auto":
            n_list = 2 ** int(np.log2(np.sqrt(embedding_length)))
            factory_str = f"IVF{n_list},PQ{embedding_size//2}x4fs"
            logger.info(f"Auto set index to {factory_str}")
            logger.info(
                f"We recommend to set n_probe to {n_list//8} for better inference performance"
            )

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
        if (self.index_train_num >= embeddings.shape[0]) or (
            self.index_train_num == -1
        ):
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype("float32")
            self.index.train(embeddings)
        else:
            selected_indices = np.random.choice(
                embeddings.shape[0],
                self.index_train_num,
                replace=False,
            )
            selected_indices = np.sort(selected_indices)
            selected_embeddings = embeddings[selected_indices].astype("float32")
            self.index.train(selected_embeddings)
        return

    def add_embeddings_batch(self, embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype("float32")
        assert self.is_trained, "Index should be trained first"
        self.index.add(embeddings)  # debug
        return

    def prepare_search_params(self, **kwargs):
        # set search kwargs
        k_factor = kwargs.get("k_factor", self.k_factor)
        n_probe = kwargs.get("n_probe", self.n_probe)
        polysemous_ht = kwargs.get("polysemous_ht", self.polysemous_ht)
        efSearch = kwargs.get("efSearch", self.efSearch)

        def get_search_params(index):
            if isinstance(index, self.faiss.IndexRefine):
                params = self.faiss.IndexRefineSearchParameters(
                    k_factor=k_factor,
                    base_index_params=get_search_params(
                        self.faiss.downcast_index(index.base_index)
                    ),
                )
            elif isinstance(index, self.faiss.IndexPreTransform):
                params = self.faiss.SearchParametersPreTransform(
                    index_params=get_search_params(
                        self.faiss.downcast_index(index.index)
                    )
                )
            elif isinstance(index, self.faiss.IndexIVFPQ):
                if hasattr(index, "quantizer"):
                    params = self.faiss.IVFPQSearchParameters(
                        nprobe=n_probe,
                        polysemous_ht=polysemous_ht,
                        quantizer_params=get_search_params(
                            self.faiss.downcast_index(index.quantizer)
                        ),
                    )
                else:
                    params = self.faiss.IVFPQSearchParameters(
                        nprobe=n_probe, polysemous_ht=polysemous_ht
                    )
            elif isinstance(index, self.faiss.IndexIVF):
                if hasattr(index, "quantizer"):
                    params = self.faiss.SearchParametersIVF(
                        nprobe=n_probe,
                        quantizer_params=get_search_params(
                            self.faiss.downcast_index(index.quantizer)
                        ),
                    )
                else:
                    params = self.faiss.SearchParametersIVF(nprobe=n_probe)
            elif isinstance(index, self.faiss.IndexHNSW):
                params = self.faiss.SearchParametersHNSW(efSearch=efSearch)
            elif isinstance(index, self.faiss.IndexPQ):
                params = self.faiss.SearchParametersPQ(polysemous_ht=polysemous_ht)
            else:
                params = None
            return params

        return get_search_params(self.index)

    def _search_batch(
        self,
        query_vectors: np.ndarray,
        top_docs: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = query_vectors.astype("float32")
        search_params = self.prepare_search_params(**search_kwargs)
        scores, indices = self.index.search(
            query_vectors, top_docs, params=search_params
        )
        return indices, scores

    def serialize(self, index_path: str = None) -> None:
        if index_path is not None:
            self.index_path = index_path
        assert self.index_path is not None, "`index_path` is not set."
        assert self.index.is_trained, "Index should be trained first."
        logger.info(f"Serializing index to {self.index_path}")
        if self.support_gpu:
            cpu_index = self.faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        self.faiss.write_index(cpu_index, self.index_path)
        return

    def deserialize(self):
        assert self.index_path is not None, "Index path is not set."
        logger.info(f"Loading index from {self.index_path}.")
        if (os.path.getsize(self.index_path) / (1024**3) > 10) and (
            not self.support_gpu
        ):
            logger.info("Index file is too large. Loading on CPU with memory map.")
            cpu_index = self.faiss.read_index(self.index_path, self.faiss.IO_FLAG_MMAP)
        else:
            cpu_index = self.faiss.read_index(self.index_path)
        self.index = self._set_index(cpu_index)
        return self.index

    def clean(self):
        if self.index is None:
            return
        if self.index_path is not None:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
        self.index.reset()
        return

    @property
    def embedding_size(self) -> int:
        if self.index is None:
            raise ValueError("Index is not initialized.")
        return self.index.d

    @property
    def is_trained(self) -> bool:
        if self.index is None:
            return False
        if isinstance(self.index, self.faiss.IndexReplicas):
            trained = True
            for i in range(self.index.count()):
                sub_index = self.faiss.downcast_index(self.index.at(i))
                if not sub_index.is_trained:
                    trained = False
            return trained
        return self.index.is_trained

    @property
    def is_addable(self) -> bool:
        return self.is_trained

    @property
    def is_flat(self) -> bool:
        def _is_flat(index) -> bool:
            if isinstance(self.index, self.faiss.IndexFlat):
                return True
            if self.support_gpu:
                if isinstance(self.index, self.faiss.GpuIndexFlat):
                    return True
            return False

        if self.index is None:
            return self.index_type == "FLAT"
        if _is_flat(self.index):
            return True
        if isinstance(self.index, self.faiss.IndexReplicas):
            all_flat = True
            for i in range(self.index.count()):
                sub_index = self.faiss.downcast_index(self.index.at(i))
                if not _is_flat(sub_index):
                    all_flat = False
            if all_flat:
                return True
        return False

    @property
    def support_gpu(self) -> bool:
        return hasattr(self.faiss, "GpuMultipleClonerOptions") and (
            len(self.device_id) > 0
        )

    def __len__(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal

    def _set_index(self, index):
        if self.support_gpu:
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
        elif len(self.device_id) > 0:
            logger.warning(
                "The installed faiss does not support GPU acceleration. "
                "Please install faiss-gpu."
            )
        return index
