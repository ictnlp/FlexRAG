import os
import re
from typing import Any, Iterable

import numpy as np

from flexrag.common import LOGGER_MANAGER, configure
from flexrag.models import EncoderProtocol

from .index_base import (
    DEFAULT_INDEX_BATCH_SIZE,
    RETRIEVER_INDEX,
    ContextIndexBase,
    DenseRawIndexBase,
    DenseRawIndexBaseConfig,
    IndexFieldsConfig,
)

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.index.scann")


@configure
class ScaNNRawIndexConfig(DenseRawIndexBaseConfig):
    """Configuration for row-level ScaNN indexing.

    :param distance_function: Vector distance or similarity function. Available
        choices are ``"IP"``, ``"L2"``, and ``"COS"``. Defaults to ``"IP"``.
    :param num_leaves: Number of leaves in the ScaNN partitioning tree.
        Defaults to 2000.
    :param num_leaves_to_search: Number of leaves searched for each query.
        Defaults to 500.
    :param num_neighbors: Number of neighbors used when building the ScaNN
        index. Defaults to 10.
    :param anisotropic_quantization_threshold: Threshold used by ScaNN's
        anisotropic quantization. Defaults to 0.2.
    :param dimensions_per_block: Number of dimensions per quantization block.
        Defaults to 2.
    :param threads: Number of ScaNN training and search threads. ``0`` keeps
        ScaNN's default thread behavior.
    :param index_train_num: Number of embeddings sampled to train the ScaNN
        tree. Values less than or equal to ``0`` mean all embeddings are used.
        Defaults to 0.
    """

    num_leaves: int = 2000
    num_leaves_to_search: int = 500
    num_neighbors: int = 10
    anisotropic_quantization_threshold: float = 0.2
    dimensions_per_block: int = 2
    threads: int = 0
    index_train_num: int = 0


class ScaNNRawIndex(DenseRawIndexBase):
    """Row-level ScaNN index."""

    config_cls = ScaNNRawIndexConfig
    cfg: ScaNNRawIndexConfig

    def __init__(
        self,
        cfg: ScaNNRawIndexConfig,
        query_encoder: EncoderProtocol,
        passage_encoder: EncoderProtocol | None = None,
    ) -> None:
        super().__init__(cfg, query_encoder, passage_encoder)
        try:
            import scann

            self.scann = scann
        except ImportError as exc:
            raise ImportError(
                "Please install scann by running `pip install scann`"
            ) from exc
        self.index = None
        return

    def build_index(
        self,
        data: Iterable[Any],
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        scratch_path: str | None = None,
    ) -> None:
        # encode the data
        self.clear()
        embeddings = self.encode_data_batch(
            data,
            is_query=False,
            batch_size=batch_size,
            scratch_path=scratch_path,
        )
        indices = list(range(len(embeddings)))

        # prepare arguments
        if self.cfg.distance_function in {"IP", "COS"}:
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
                anisotropic_quantization_threshold=(
                    self.cfg.anisotropic_quantization_threshold
                ),
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
            docids=indices,
            database=embeddings,
            batch_size=len(embeddings),
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
        scores = self._postprocess_scores(scores)
        return indices, scores

    def save_to_local(self, index_path: str) -> None:
        assert self.is_trained, "Index should be trained before saving."
        self._save_config(index_path)
        self.index.serialize(index_path)
        return

    def _load_from_local(self, index_path: str) -> None:
        try:
            self._update_assets(index_path)
            self.index = self.scann.scann_ops_pybind.load_searcher(index_path)
        except Exception as exc:
            raise FileNotFoundError(f"Unable to load index from {index_path}") from exc
        return

    def clear(self) -> None:
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
        file_path = os.path.join(index_path, "scann_assets.pbtxt")
        if not os.path.exists(file_path):
            logger.error(
                f"Asset file (scann_assets.pbtxt) not found. "
                f"Please check the `index_path` ({index_path})."
            )
        new_lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r"(?:\s*asset_path:\s+\")([^\"]+)(?:\")", line)
                if match:
                    asset_name = os.path.basename(match.group(1))
                    new_path = os.path.join(index_path, asset_name)
                    assert os.path.exists(new_path), (
                        f"Asset {asset_name} not found at {new_path}"
                    )
                    line = re.sub(
                        r"(asset_path:\s+\")[^\"]+(\")",
                        f"\\1{new_path}\\2",
                        line,
                    )
                new_lines.append(line)
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return

    def __len__(self) -> int:
        if self.index is None:
            return 0
        return self.index.size()


@configure
class ScaNNIndexConfig(ScaNNRawIndexConfig, IndexFieldsConfig):
    """Configuration for context-level ScaNNIndex.

    :param indexed_fields: Context fields to index. If ``None``, all fields are
        indexed. Defaults to ``None``.
    :param merge_method: How to merge multiple field-level scores for the same
        context. Available choices are ``"max"``, ``"sum"``, ``"mean"``, and
        ``"concat"``. Defaults to ``"max"``.
    :param distance_function: Vector distance or similarity function. Available
        choices are ``"IP"``, ``"L2"``, and ``"COS"``. Defaults to ``"IP"``.
    :param num_leaves: Number of leaves in the ScaNN partitioning tree.
        Defaults to 2000.
    :param num_leaves_to_search: Number of leaves searched for each query.
        Defaults to 500.
    :param num_neighbors: Number of neighbors used when building the ScaNN
        index. Defaults to 10.
    :param anisotropic_quantization_threshold: Threshold used by ScaNN's
        anisotropic quantization. Defaults to 0.2.
    :param dimensions_per_block: Number of dimensions per quantization block.
        Defaults to 2.
    :param threads: Number of ScaNN training and search threads. ``0`` keeps
        ScaNN's default thread behavior.
    :param index_train_num: Number of embeddings sampled to train the ScaNN
        tree. Values less than or equal to ``0`` mean all embeddings are used.
        Defaults to 0.
    """


@RETRIEVER_INDEX("scann", config_class=ScaNNIndexConfig)
class ScaNNIndex(ContextIndexBase):
    """Context-level ScaNN index."""

    raw_index_cls = ScaNNRawIndex
    raw_config_cls = ScaNNRawIndexConfig
    config_cls = ScaNNIndexConfig

    @property
    def query_encoder(self):
        return self.raw_index.query_encoder

    @property
    def passage_encoder(self):
        return self.raw_index.passage_encoder
