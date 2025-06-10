import os
import re
import shutil
from copy import deepcopy
from typing import Any, Iterable

import numpy as np

from flexrag.models import ENCODERS
from flexrag.utils import LOGGER_MANAGER, configure
from flexrag.utils.configure import extract_config

from .index_base import RETRIEVER_INDEX, DenseIndexBase, DenseIndexBaseConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.index.scann")


@configure
class ScaNNIndexConfig(DenseIndexBaseConfig):
    """The configuration for the `ScaNNIndex`.

    :param num_leaves: The number of leaves in the tree. Defaults to 2000.
    :type num_leaves: int
    :param num_leaves_to_search: The number of leaves to search. Defaults to 500.
    :type num_leaves_to_search: int
    :param num_neighbors: The number of neighbors to search. Defaults to 10.
    :type num_neighbors: int
    :param anisotropic_quantization_threshold: The anisotropic quantization threshold. Defaults to 0.2.
    :type anisotropic_quantization_threshold: float
    :param dimensions_per_block: The number of dimensions per block. Defaults to 2.
    :type dimensions_per_block: int
    :param threads: The number of threads to use. Defaults to 0 (auto).
    :type threads: int
    :param index_train_num: The number of samples to train the index. Defaults to 0 (all).
    :type index_train_num: int
    """

    num_leaves: int = 2000
    num_leaves_to_search: int = 500
    num_neighbors: int = 10
    anisotropic_quantization_threshold: float = 0.2
    dimensions_per_block: int = 2
    threads: int = 0
    index_train_num: int = 0


@RETRIEVER_INDEX("scann", config_class=ScaNNIndexConfig)
class ScaNNIndex(DenseIndexBase):
    """ScaNNIndex is a wrapper for the `ScaNN <https://github.com/google-research/google-research/tree/master/scann>`_ library.

    ScaNNIndex runs on CPUs with both high speed and accuracy.
    However, it requires more memory than ``FaissIndex``.
    """

    def __init__(self, cfg: ScaNNIndexConfig) -> None:
        super().__init__(cfg)
        self.cfg = extract_config(cfg, ScaNNIndexConfig)
        # check scann
        try:
            import scann

            self.scann = scann
        except:
            raise ImportError("Please install scann by running `pip install scann`")

        # load the index if index_path is provided
        if self.cfg.index_path is not None:
            if os.path.exists(self.cfg.index_path):
                logger.info(f"Loading index from {self.cfg.index_path}.")
                try:
                    self._update_assets(self.cfg.index_path)
                    self.index = self.scann.scann_ops_pybind.load_searcher(
                        self.cfg.index_path
                    )
                except:
                    raise FileNotFoundError(
                        f"Unable to load index from {self.cfg.index_path}"
                    )
        else:
            self.index = None
        return

    def build_index(self, data: Iterable[Any]) -> None:
        # encode the data
        self.clear()
        embeddings = self.encode_data_batch(data, is_query=False)
        indices = list(range(len(embeddings)))

        # prepare arguments
        if self.cfg.distance_function == "IP":
            distance_measure = "dot_product"
        elif self.cfg.distance_function == "COS":
            distance_measure = "dot_product"
        else:
            distance_measure = "squared_l2"
        train_num = (
            len(embeddings)
            if self.cfg.index_train_num <= 0
            else self.cfg.index_train_num
        )

        # prepare the builder
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

        # build the index
        self.index = builder.build(indices)
        self.index.set_num_threads(self.cfg.threads)

        # clear the memmap
        if isinstance(embeddings, np.memmap):
            os.remove(embeddings.filename)
            del embeddings
        return

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype("float32")
        assert self.is_trained, "Index should be trained first"
        indices = list(range(self.index.size(), self.index.size() + len(embeddings)))
        self.index.upsert(
            docids=indices, database=embeddings, batch_size=self.cfg.batch_size
        )
        return

    def search(
        self,
        query: list[Any],
        top_k: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = self.encode_data(query, is_query=True)
        indices, scores = self.index.search_batched(
            query_vectors, top_k, **search_kwargs
        )
        indices = np.array(indices)
        return indices, scores

    def save_to_local(self, index_path: str = None) -> None:
        # check if the index is serializable
        if index_path is not None:
            self.cfg.index_path = index_path
        assert self.cfg.index_path is not None, "`index_path` is not set."
        assert self.is_trained, "Index should be trained before saving."
        if not os.path.exists(index_path):
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
        self.index.serialize(self.cfg.index_path)
        return

    def clear(self):
        if not self.is_trained:
            return
        if self.cfg.index_path is not None:
            if os.path.exists(self.cfg.index_path):
                shutil.rmtree(self.cfg.index_path)
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

    @property
    def is_addable(self) -> bool:
        return self.is_trained

    def _update_assets(self, index_path: str) -> None:
        """As the `ScaNN` requires the assets table to find the index files,
        we need to update the path in the `scann_assets.pbtxt` file.

        :param index_path: The path to the index.
        :type index_path: str
        :return: None
        :rtype: None
        """
        file_path = os.path.join(index_path, "scann_assets.pbtxt")
        if not os.path.exists(file_path):
            logger.error(
                f"Asset file (scann_assets.pbtxt) not found. "
                f"Please check the `index_path` ({index_path})."
            )
        new_lines = []
        with open(os.path.join(index_path, "scann_assets.pbtxt"), "r") as f:
            for line in f:
                match = re.match(r"(?:\s*asset_path:\s+\")([^\"]+)(?:\")", line)
                if match:
                    asset_name = os.path.basename(match.group(1))
                    new_path = os.path.join(index_path, asset_name)
                    assert os.path.exists(
                        new_path
                    ), f"Asset {asset_name} not found at {new_path}"
                    line = re.sub(
                        r"(asset_path:\s+\")[^\"]+(\")",
                        f"\\1{new_path}\\2",
                        line,
                    )
                    new_lines.append(line)
                else:
                    new_lines.append(line)
        with open(os.path.join(index_path, "scann_assets.pbtxt"), "w") as f:
            f.writelines(new_lines)
        return

    def __len__(self) -> int:
        if self.index is None:
            return 0
        return self.index.size()
