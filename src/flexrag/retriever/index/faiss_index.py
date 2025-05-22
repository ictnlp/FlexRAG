import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import faiss
import numpy as np
from omegaconf import OmegaConf

from flexrag.utils import LOGGER_MANAGER, Choices

from .index_base import RETRIEVER_INDEX, DenseIndexBase, DenseIndexBaseConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retriever.index.faiss")


@dataclass
class FaissIndexConfig(DenseIndexBaseConfig):
    """The configuration for the `FaissIndex`.

    :param index_type: Building param: the type of the index. Defaults to "auto".
        available choices are "FLAT", "IVF", "PQ", "IVFPQ", and "auto".
        If set to "auto", the index will be set to "IVF{n_list},PQ{embedding_size//2}x4fs".
    :type index_type: str
    :param n_subquantizers: Building param: the number of subquantizers. Defaults to 8.
        This parameter is only used when the index type is "PQ" or "IVFPQ".
    :type n_subquantizers: int
    :param n_bits: Building param: the number of bits per subquantizer. Defaults to 8.
        This parameter is only used when the index type is "PQ" or "IVFPQ".
    :type n_bits: int
    :param n_list: Building param: the number of cells. Defaults to 1000.
        This parameter is only used when the index type is "IVF" or "IVFPQ".
    :type n_list: int
    :param factory_str: Building param: the factory string to build the index. Defaults to None.
        If set, the `index_type` will be ignored.
    :type factory_str: Optional[str]
    :param index_train_num: Building param: the number of data used to train the index. Defaults to -1.
        If set to -1, all data will be used to train the index.
    :type index_train_num: int
    :param n_probe: Inference param: the number of probes. Defaults to None.
        If not set, the number of probes will be set to `n_list // 8`.
        This parameter is only used when the index type is "IVF" or "IVFPQ".
    :type n_probe: Optional[int]
    :param device_id: Inference param: the device(s) to use. Defaults to [].
        [] means CPU. If set, the index will be accelerated with GPU.
    :type device_id: list[int]
    :param k_factor: Inference param: the k factor for search. Defaults to 10.
    :type k_factor: int
    :param polysemous_ht: Inference param: the polysemous hash table. Defaults to 0.
    :type polysemous_ht: int
    :param efSearch: Inference param: the efSearch for HNSW. Defaults to 100.
    :type efSearch: int
    """

    index_type: Choices(["FLAT", "IVF", "PQ", "IVFPQ", "auto"]) = "auto"  # type: ignore
    n_subquantizers: int = 8
    n_bits: int = 8
    n_list: int = 1000
    factory_str: Optional[str] = None
    index_train_num: int = -1
    # Inference Arguments
    n_probe: Optional[int] = None
    device_id: list[int] = field(default_factory=list)
    k_factor: int = 10
    polysemous_ht: int = 0
    efSearch: int = 100


@RETRIEVER_INDEX("faiss", config_class=FaissIndexConfig)
class FaissIndex(DenseIndexBase):
    """FaissIndex employs `faiss <https://github.com/facebookresearch/faiss>`_ library to build and search indexes with embeddings.
    FaissIndex supports both CPU and GPU acceleration.
    FaissIndex supports various index types, including FLAT, IVF, PQ, IVFPQ, and auto.
    FaissIndex provides a flexible and efficient way to build and search indexes with embeddings.
    """

    cfg: FaissIndexConfig

    def __init__(self, cfg: FaissIndexConfig) -> None:
        super().__init__(cfg)
        self.cfg = FaissIndexConfig.extract(cfg)
        # prepare index
        self.index = None

        # load the index if index_path is provided
        if self.cfg.index_path is not None:
            if os.path.exists(self.cfg.index_path):
                logger.info(f"Loading index from {self.cfg.index_path}")
                try:
                    index_path = os.path.join(self.cfg.index_path, "index.faiss")
                    cpu_index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
                    self.index = self._set_index(cpu_index)
                except:
                    raise FileNotFoundError(
                        f"Unable to load index from {self.cfg.index_path}"
                    )
        return

    def build_index(self, data: Iterable[Any]) -> None:
        self.clear()
        embeddings = self.encode_data_batch(data, is_query=False)
        self.index = self._prepare_index(
            index_type=self.cfg.index_type,
            distance_function=self.cfg.distance_function,
            embedding_size=embeddings.shape[1],
            embedding_length=embeddings.shape[0],
            n_list=self.cfg.n_list,
            n_subquantizers=self.cfg.n_subquantizers,
            n_bits=self.cfg.n_bits,
            factory_str=self.cfg.factory_str,
        )
        self._train_index(embeddings)
        self.add_embeddings_batch(embeddings)
        if isinstance(embeddings, np.memmap):
            emb_path = embeddings.filename
            os.remove(emb_path)
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
                basic_index = faiss.IndexFlatIP(embedding_size)
                basic_metric = faiss.METRIC_INNER_PRODUCT
            case "L2":
                basic_index = faiss.IndexFlatL2(embedding_size)
                basic_metric = faiss.METRIC_L2
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
            index = faiss.index_factory(
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
                    index = faiss.IndexIVFFlat(
                        basic_index,
                        embedding_size,
                        n_list,
                        basic_metric,
                    )
                case "PQ":
                    index = faiss.IndexPQ(
                        embedding_size,
                        n_subquantizers,
                        n_bits,
                    )
                case "IVFPQ":
                    index = faiss.IndexIVFPQ(
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

    def _train_index(self, embeddings: np.ndarray) -> None:
        if self.is_trained:
            logger.info("Index is trained already.")
            return
        logger.info("Training index")
        if (self.cfg.index_train_num >= embeddings.shape[0]) or (
            self.cfg.index_train_num == -1
        ):
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype("float32")
            self.index.train(embeddings)
        else:
            selected_indices = np.random.choice(
                embeddings.shape[0],
                self.cfg.index_train_num,
                replace=False,
            )
            selected_indices = np.sort(selected_indices)
            selected_embeddings = embeddings[selected_indices].astype("float32")
            self.index.train(selected_embeddings)
        return

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype("float32")
        assert self.is_trained, "Index should be trained first"
        self.index.add(embeddings)
        return

    def _prepare_search_params(self, **kwargs):
        """A helper function to prepare search parameters for the index.

        :return: The search parameters for the index.
        :rtype: faiss.SearchParameters
        """
        # set search kwargs
        k_factor = kwargs.get("k_factor", self.cfg.k_factor)
        n_probe = kwargs.get("n_probe", self.cfg.n_probe)
        if n_probe is None:
            n_probe = getattr(self.index, "nlist", 256) // 8
        polysemous_ht = kwargs.get("polysemous_ht", self.cfg.polysemous_ht)
        efSearch = kwargs.get("efSearch", self.cfg.efSearch)

        def get_search_params(index):
            if isinstance(index, faiss.IndexRefine):
                params = faiss.IndexRefineSearchParameters(
                    k_factor=k_factor,
                    base_index_params=get_search_params(
                        faiss.downcast_index(index.base_index)
                    ),
                )
            elif isinstance(index, faiss.IndexPreTransform):
                params = faiss.SearchParametersPreTransform(
                    index_params=get_search_params(faiss.downcast_index(index.index))
                )
            elif isinstance(index, faiss.IndexIVFPQ):
                if hasattr(index, "quantizer"):
                    params = faiss.IVFPQSearchParameters(
                        nprobe=n_probe,
                        polysemous_ht=polysemous_ht,
                        quantizer_params=get_search_params(
                            faiss.downcast_index(index.quantizer)
                        ),
                    )
                else:
                    params = faiss.IVFPQSearchParameters(
                        nprobe=n_probe, polysemous_ht=polysemous_ht
                    )
            elif isinstance(index, faiss.IndexIVF):
                if hasattr(index, "quantizer"):
                    params = faiss.SearchParametersIVF(
                        nprobe=n_probe,
                        quantizer_params=get_search_params(
                            faiss.downcast_index(index.quantizer)
                        ),
                    )
                else:
                    params = faiss.SearchParametersIVF(nprobe=n_probe)
            elif isinstance(index, faiss.IndexHNSW):
                params = faiss.SearchParametersHNSW(efSearch=efSearch)
            elif isinstance(index, faiss.IndexPQ):
                params = faiss.SearchParametersPQ(polysemous_ht=polysemous_ht)
            else:
                params = None
            return params

        return get_search_params(self.index)

    def search(
        self,
        query: list[Any],
        top_k: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = self.encode_data(query, is_query=True)
        search_params = self._prepare_search_params(**search_kwargs)
        scores, indices = self.index.search(query_vectors, top_k, params=search_params)
        return indices, scores

    def save_to_local(self, index_path: str = None) -> None:
        # check if the index is serializable
        if index_path is not None:
            self.cfg.index_path = index_path
        assert self.cfg.index_path is not None, "`index_path` is not set."
        assert self.index.is_trained, "Index should be trained first."
        if not os.path.exists(index_path):
            os.makedirs(self.cfg.index_path)
        logger.info(f"Serializing index to {self.cfg.index_path}")

        # save the configuration
        config_path = os.path.join(self.cfg.index_path, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            OmegaConf.save(self.cfg, f)
        id_path = os.path.join(self.cfg.index_path, "cls.id")
        with open(id_path, "w", encoding="utf-8") as f:
            f.write(self.__class__.__name__)

        # serialize the index
        index_path = os.path.join(index_path, "index.faiss")
        if self.support_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        faiss.write_index(cpu_index, index_path)
        return

    def clear(self):
        if self.index is None:
            return
        self.index.reset()

        if self.cfg.index_path is not None:
            if os.path.exists(self.cfg.index_path):
                shutil.rmtree(self.cfg.index_path)
        return

    @property
    def embedding_size(self) -> int:
        if self.index is not None:
            return self.index.d
        if self.passage_encoder is not None:
            return self.passage_encoder.embedding_size
        if self.query_encoder is not None:
            return self.query_encoder.embedding_size
        raise ValueError("Index is not initialized.")

    @property
    def is_trained(self) -> bool:
        if self.index is None:
            return False
        return self.index.is_trained

    @property
    def is_addable(self) -> bool:
        return self.is_trained

    @property
    def support_gpu(self) -> bool:
        return hasattr(faiss, "GpuMultipleClonerOptions") and (
            len(self.cfg.device_id) > 0
        )

    def _set_index(self, index):
        if self.support_gpu:
            logger.info("Accelerating index with GPU.")
            option = faiss.GpuMultipleClonerOptions()
            option.useFloat16 = True
            option.shard = True
            if isinstance(index, faiss.IndexIVFFlat):
                option.common_ivf_quantizer = True
            index = faiss.index_cpu_to_gpus_list(
                index,
                co=option,
                gpus=self.cfg.device_id,
                ngpu=len(self.cfg.device_id),
            )
        elif len(self.cfg.device_id) > 0:
            logger.warning(
                "The installed faiss does not support GPU acceleration. "
                "Please install faiss-gpu."
            )
        return index

    def __len__(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal
