import os
import shutil
from copy import deepcopy
from typing import Any, Iterable, Optional

import faiss
import numpy as np

from flexrag.common import LOGGER_MANAGER, configure
from flexrag.common.configure import extract_config
from flexrag.models import ENCODERS

from .index_base import RETRIEVER_INDEX, DenseIndexBase, DenseIndexBaseConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retriever.index.faiss")


@configure
class FaissIndexConfig(DenseIndexBaseConfig):
    """The configuration for the `FaissIndex`.

    :param factory_str: Building param: the factory string to build the index. Defaults to None.
        If set to None, the index will be chosen automatically based on the corpus size.
    :type factory_str: Optional[str]
    :param index_train_num: Building param: the number of data used to train the index. Defaults to -1.
        If set to -1, all data will be used to train the index.
    :type index_train_num: int
    :param n_probe: Inference param: the number of probes. Defaults to None.
        If not set, the number of probes will be set to `index.nlist // 8` when the
        resolved index contains an IVF component.
    :type n_probe: Optional[int]
    :param k_factor: Inference param: the k factor for search. Defaults to 10.
    :type k_factor: int
    :param polysemous_ht: Inference param: the polysemous hash table. Defaults to 0.
    :type polysemous_ht: int
    :param efSearch: Inference param: the efSearch for HNSW. Defaults to 100.
    :type efSearch: int
    """

    factory_str: Optional[str] = None
    index_train_num: int = -1
    # Inference Arguments
    n_probe: Optional[int] = None
    k_factor: int = 10
    polysemous_ht: int = 0
    efSearch: int = 100


@RETRIEVER_INDEX("faiss", config_class=FaissIndexConfig)
class FaissIndex(DenseIndexBase):
    """FaissIndex employs `faiss <https://github.com/facebookresearch/faiss>`_ library to build and search indexes with embeddings.
    FaissIndex runs on CPU-backed Faiss indexes.
    FaissIndex supports both automatic index selection and explicit Faiss factory strings.
    FaissIndex provides a flexible and efficient way to build and search indexes with embeddings.
    """

    cfg: FaissIndexConfig

    def __init__(self, cfg: FaissIndexConfig) -> None:
        super().__init__(cfg)
        self.cfg = extract_config(cfg, FaissIndexConfig)
        # prepare index
        self.index = None

        # load the index if index_path is provided
        if self.cfg.index_path is not None:
            if os.path.exists(self.cfg.index_path):
                logger.info(f"Loading index from {self.cfg.index_path}")
                try:
                    index_path = os.path.join(self.cfg.index_path, "index.faiss")
                    self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
                except:
                    raise FileNotFoundError(
                        f"Unable to load index from {self.cfg.index_path}"
                    )
        return

    def build_index(self, data: Iterable[Any]) -> None:
        self.clear()
        embeddings = self.encode_data_batch(data, is_query=False)
        factory_str = self._resolve_factory_str(
            embedding_size=embeddings.shape[1],
            embedding_length=embeddings.shape[0],
            factory_str=self.cfg.factory_str,
        )
        self.index = self._prepare_index(
            distance_function=self.cfg.distance_function,
            embedding_size=embeddings.shape[1],
            factory_str=factory_str,
        )
        self._train_index(embeddings)
        self.add_embeddings_batch(embeddings)
        if isinstance(embeddings, np.memmap):
            emb_path = embeddings.filename
            os.remove(emb_path)
        return

    def _resolve_factory_str(
        self,
        embedding_size: int,  # the dimension of the embeddings
        embedding_length: int,  # the number of the embeddings
        factory_str: Optional[str] = None,
    ) -> str:
        if factory_str is not None:
            logger.info(f"Using Faiss factory string: {factory_str}")
            return factory_str

        n = embedding_length
        d = embedding_size
        raw_bytes = n * d * np.dtype(np.float32).itemsize
        n_list = max(64, 2 ** int(np.log2(np.sqrt(n))))

        if n < 10_000 or (n * d <= 20_000_000 and raw_bytes <= 128 * 1024 * 1024):
            resolved_factory_str = "Flat"
            logger.info("Auto set index to Flat")
            return resolved_factory_str

        if d <= 1024 and n <= 300_000:
            if d <= 256:
                hnsw_m = 32
            elif d <= 768:
                hnsw_m = 24
            else:
                hnsw_m = 16
            resolved_factory_str = f"HNSW{hnsw_m}"
            logger.info(f"Auto set index to {resolved_factory_str}")
            return resolved_factory_str

        if n <= 1_000_000 and raw_bytes <= 8 * 1024 * 1024 * 1024:
            resolved_factory_str = f"IVF{n_list},Flat"
            logger.info(f"Auto set index to {resolved_factory_str}")
            logger.info(
                f"We recommend to set n_probe to {n_list//8} "
                f"for better inference performance."
            )
            return resolved_factory_str

        pq_m = None
        for m in range(min(64, d // 2), 7, -1):
            if d % m == 0:
                pq_m = m
                break
        if pq_m is None:
            resolved_factory_str = f"IVF{n_list},Flat"
            logger.warning(
                "Unable to derive a suitable PQ configuration for embedding size "
                f"{d}. Falling back to {resolved_factory_str}."
            )
            logger.info(
                f"We recommend to set n_probe to {n_list//8} "
                f"for better inference performance."
            )
            return resolved_factory_str

        resolved_factory_str = f"IVF{n_list},PQ{pq_m}x4fs"
        logger.info(f"Auto set index to {resolved_factory_str}")
        logger.info(
            f"We recommend to set n_probe to {n_list//8} "
            f"for better inference performance."
        )
        return resolved_factory_str

    def _prepare_index(
        self,
        distance_function: str,
        embedding_size: int,  # the dimension of the embeddings
        factory_str: str,
    ):
        # prepare distance function
        match distance_function:
            case "IP" | "COS":
                basic_metric = faiss.METRIC_INNER_PRODUCT
            case "L2":
                basic_metric = faiss.METRIC_L2
            case _:
                raise ValueError(f"Unknown distance function: {distance_function}")

        index = faiss.index_factory(
            embedding_size,
            factory_str,
            basic_metric,
        )
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
        scores = self._postprocess_scores(scores)
        return indices, scores

    def save_to_local(self, index_path: str = None) -> None:
        # check if the index is serializable
        if index_path is not None:
            self.cfg.index_path = index_path
        assert self.cfg.index_path is not None, "`index_path` is not set."
        assert self.index.is_trained, "Index should be trained first."
        if not os.path.exists(self.cfg.index_path):
            os.makedirs(self.cfg.index_path)
        logger.info(f"Serializing index to {self.cfg.index_path}")

        # save the configuration
        cfg = deepcopy(self.cfg)
        cfg.query_encoder_config = ENCODERS.squeeze(cfg.query_encoder_config)
        cfg.passage_encoder_config = ENCODERS.squeeze(cfg.passage_encoder_config)
        cfg.index_path = ""
        config_path = os.path.join(self.cfg.index_path, "config.yaml")
        cfg.dump(config_path)
        id_path = os.path.join(self.cfg.index_path, "cls.id")
        with open(id_path, "w", encoding="utf-8") as f:
            f.write(self.__class__.__name__)

        # serialize the index
        index_path = os.path.join(self.cfg.index_path, "index.faiss")
        faiss.write_index(self.index, index_path)
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

    def __len__(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal
