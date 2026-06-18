import os
import pickle
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Annotated, Any, Generator, Iterable, Optional
from uuid import uuid4

import numpy as np

from flexrag.common import (
    FLEXRAG_CACHE_DIR,
    LOGGER_MANAGER,
    Choices,
    ProgressDisplay,
    Register,
    SimpleProgressLogger,
    configure,
    trace,
)
from flexrag.common.configure import extract_config
from flexrag.models.encoders import EncoderProtocol

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.index")


DEFAULT_INDEX_BATCH_SIZE = 512


class RawIndexBase(ABC):
    """Base class for row-level indexes.

    Raw indexes index flat data rows and return row ids from ``search``. They do
    not own a configured local path; callers pass an explicit path when saving or
    loading raw artifacts.
    """

    config_cls: type[Any]
    cfg: Any

    def __init__(self, cfg: Any, **_: Any) -> None:
        self.cfg = extract_config(cfg, self.config_cls)
        return

    @abstractmethod
    def build_index(
        self,
        data: Iterable[Any],
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        scratch_path: str | None = None,
    ) -> None:
        """Build the raw index from flat row data.

        Raw indexes do not receive context IDs and must preserve the input row
        order as the row ID space returned by :meth:`search`.

        :param data: Flat data rows to index.
        :param batch_size: Runtime batch size used by implementations that
            encode or add data in batches. Defaults to
            :data:`DEFAULT_INDEX_BATCH_SIZE`.
        :param scratch_path: Optional directory for temporary build artifacts.
        :return: None.
        """
        return

    def insert_batch(
        self,
        data: Iterable[Any],
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        """Insert flat row data in batches.

        This helper calls :meth:`insert` for each batch and does not persist the
        index. Callers that own a storage path are responsible for saving after
        mutation.

        :param data: Flat rows to insert.
        :param batch_size: Runtime batch size used for insertion. Defaults to
            :data:`DEFAULT_INDEX_BATCH_SIZE`.
        :param log_interval: Number of inserted rows between progress updates.
        :param display: Progress display mode.
        :return: None.
        """
        assert self.is_addable, "Current index is not addable."

        def get_data_batch() -> Generator[list[Any], None, None]:
            batch = []
            for item in data:
                batch.append(item)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
            return

        with SimpleProgressLogger(
            logger, interval=log_interval, display=display
        ) as p_logger:
            for batch in get_data_batch():
                self.insert(batch)
                p_logger.update(step=len(batch), desc="Adding data")
        return

    @abstractmethod
    def insert(self, data: list[Any]) -> None:
        """Insert one batch of flat row data.

        Implementations may raise ``NotImplementedError`` when the underlying
        index does not support incremental insertion.

        :param data: Flat rows to append to the raw index.
        :return: None.
        """
        return

    @trace("retriever.raw_index.search")
    def search_batch(
        self,
        query: Iterable[Any],
        top_k: int = 10,
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search flat queries in batches.

        :param query: Query items accepted by the raw index implementation.
        :param top_k: Number of row-level hits to return per query.
        :param batch_size: Runtime batch size used for searching. Defaults to
            :data:`DEFAULT_INDEX_BATCH_SIZE`.
        :param log_interval: Number of queried items between progress updates.
        :param display: Progress display mode.
        :param search_kwargs: Extra search options forwarded to :meth:`search`.
        :return: A pair ``(row_indices, scores)`` with one row per query.
        """

        def get_batch():
            batch = []
            for item in query:
                batch.append(item)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
            return

        scores = []
        indices = []
        total = len(query) if hasattr(query, "__len__") else None
        with SimpleProgressLogger(
            logger, total, interval=log_interval, display=display
        ) as p_logger:
            for q in get_batch():
                r = self.search(q, top_k, **search_kwargs)
                indices.append(r[0])
                scores.append(r[1])
                p_logger.update(step=len(q), desc="Searching")
        return np.concatenate(indices, axis=0), np.concatenate(scores, axis=0)

    @abstractmethod
    def search(
        self,
        query: list[Any],
        top_k: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the raw index and return row-level results.

        :param query: Query items accepted by the raw index implementation.
        :param top_k: Number of row-level hits to return per query.
        :param search_kwargs: Implementation-specific search options.
        :return: A pair ``(row_indices, scores)``. ``row_indices`` contains raw
            integer row IDs, not context IDs.
        """
        return

    @property
    @abstractmethod
    def is_addable(self) -> bool:
        """Whether the raw index supports incremental insertion.

        :return: ``True`` if :meth:`insert` can append data without rebuilding.
        """
        return

    def _save_config(self, index_path: str) -> None:
        os.makedirs(index_path, exist_ok=True)
        self.cfg.dump(os.path.join(index_path, "config.yaml"))
        return

    @abstractmethod
    def save_to_local(self, index_path: str) -> None:
        """Save raw index artifacts to a concrete directory.

        Raw index paths are explicit call arguments rather than config fields.
        The caller owns the lifecycle of the directory.

        :param index_path: Directory where raw config and artifacts are saved.
        :return: None.
        """
        return

    @classmethod
    def load_from_local(cls, index_path: str, **kwargs) -> "RawIndexBase":
        """Load raw index artifacts from a concrete directory.

        :param index_path: Directory containing raw config and artifacts.
        :param kwargs: Extra constructor dependencies, such as dense encoders.
        :return: Loaded raw index instance.
        """
        assert os.path.exists(index_path), f"Index path {index_path} does not exist."
        config_path = os.path.join(index_path, "config.yaml")
        assert os.path.exists(config_path), (
            f"Configuration file {config_path} does not exist."
        )
        cfg = cls.config_cls.load(config_path)
        index = cls(cfg, **kwargs)
        index._load_from_local(index_path)
        return index

    @abstractmethod
    def _load_from_local(self, index_path: str) -> None:
        return

    @abstractmethod
    def clear(self) -> None:
        """Clear in-memory raw index state without deleting disk artifacts.

        :return: None.
        """
        return

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of flat rows in the raw index.

        :return: Number of indexed raw rows.
        """
        return

    @property
    @abstractmethod
    def infimum(self) -> float:
        """Return the lower bound of raw index scores.

        :return: Minimum possible score used for score normalization.
        """
        return

    @property
    @abstractmethod
    def supremum(self) -> float:
        """Return the upper bound of raw index scores.

        :return: Maximum possible score used for score normalization.
        """
        return


@configure
class IndexFieldsConfig:
    """Configuration for projecting context fields into an index.

    :param indexed_fields: Context fields to index. If ``None``, all fields in
        each context are indexed. Missing fields are skipped. Defaults to
        ``None``.
    :param merge_method: How to merge multiple field-level scores belonging to
        the same context. Available choices are ``"max"``, ``"sum"``,
        ``"mean"``, and ``"concat"``. Defaults to ``"max"``.
    """

    indexed_fields: Optional[list[str]] = None
    merge_method: Annotated[str, Choices("max", "sum", "mean", "concat")] = "max"


class ContextIndexBase(ABC):
    """Base class for retriever-facing context-level indexes.

    Context indexes own the logical index lifecycle. They project context
    dictionaries into flat raw-index rows, keep the row/context mapping, and
    return context IDs from search.
    """

    raw_index_cls: type[RawIndexBase]
    raw_config_cls: type[Any]
    config_cls: type[IndexFieldsConfig] = IndexFieldsConfig
    cfg: IndexFieldsConfig

    def __init__(self, cfg: IndexFieldsConfig, **raw_kwargs) -> None:
        """Create an in-memory context-level index.

        :param cfg: Context index configuration.
        :param raw_kwargs: Dependencies forwarded to the raw index constructor,
            such as dense query and passage encoders.
        """
        self.cfg = extract_config(cfg, self.config_cls)
        raw_cfg = extract_config(self.cfg, self.raw_config_cls)
        self.raw_index = self.raw_index_cls(raw_cfg, **raw_kwargs)
        self._reset_mapping()
        self._check_mapping_consistency()
        return

    def _reset_mapping(self) -> None:
        self.index_to_context_id: dict[int, str] = {}
        self.context_id_to_index: dict[str, list[int]] = defaultdict(list)
        self.max_field_num = 1
        return

    def _check_mapping_consistency(self) -> None:
        assert len(self.index_to_context_id) == len(self.raw_index), (
            "The length of the raw index and the context-id mapping should be the same."
        )
        return

    def _load_state_from_local(self, index_path: str, **raw_kwargs) -> None:
        raw_path = os.path.join(index_path, "raw")
        if os.path.exists(os.path.join(raw_path, "config.yaml")):
            self.raw_index = self.raw_index_cls.load_from_local(
                raw_path,
                **raw_kwargs,
            )

        mapping_path = os.path.join(index_path, "context_mapping.pkl")
        if os.path.exists(mapping_path):
            with open(mapping_path, "rb") as f:
                mapping = pickle.load(f)
            self.context_id_to_index = defaultdict(
                list,
                mapping["context_id_to_index"],
            )
            self.index_to_context_id = mapping["index_to_context_id"]
            self.max_field_num = mapping["max_field_num"]
        else:
            assert len(self.raw_index) == 0, (
                "The raw index should be empty before building context mapping."
            )
            self._reset_mapping()
        return

    def _iter_index_data(
        self,
        context_ids: Iterable[str],
        data: Iterable[dict[str, Any]],
    ) -> Generator[tuple[str, Any], None, None]:
        for context_id, item in zip(context_ids, data):
            if self.cfg.indexed_fields is None:
                indexed_fields = list(item.keys())
            else:
                indexed_fields = [
                    field for field in self.cfg.indexed_fields if field in item
                ]

            if self.cfg.merge_method == "concat":
                concat_text = ""
                for field in indexed_fields:
                    assert isinstance(item[field], str)
                    concat_text += f"{field}: {item[field]} "
                yield context_id, concat_text
            else:
                self.max_field_num = max(self.max_field_num, len(indexed_fields))
                for field in indexed_fields:
                    yield context_id, item[field]
        return

    def build_index(
        self,
        context_ids: Iterable[str],
        data: Iterable[dict[str, Any]],
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        scratch_path: str | None = None,
    ) -> None:
        """Build the context index from context IDs and context data.

        This resets the raw index and all row/context mappings. It only updates
        in-memory state; call :meth:`save_to_local` explicitly to persist.

        :param context_ids: Context IDs corresponding to ``data``.
        :param data: Context dictionaries to project into raw-index rows.
        :param batch_size: Runtime batch size forwarded to the underlying raw
            index build. Defaults to :data:`DEFAULT_INDEX_BATCH_SIZE`.
        :param scratch_path: Optional directory for temporary build artifacts,
            such as dense embedding memmaps.
        :return: None.
        """
        self.raw_index.clear()
        self._reset_mapping()
        row_context_ids: list[str] = []

        def get_data() -> Generator[Any, None, None]:
            for context_id, item in self._iter_index_data(context_ids, data):
                row_context_ids.append(context_id)
                yield item
            return

        self.raw_index.build_index(
            get_data(),
            batch_size=batch_size,
            scratch_path=scratch_path,
        )
        for idx, context_id in enumerate(row_context_ids):
            self.context_id_to_index[context_id].append(idx)
            self.index_to_context_id[idx] = context_id

        return

    def search_batch(
        self,
        query: list[Any],
        top_k: int,
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
        **search_kwargs,
    ) -> tuple[list[list[str]], np.ndarray]:
        """Search queries in batches and return context-level results.

        :param query: Query items accepted by the underlying raw index.
        :param top_k: Number of context-level hits to return per query.
        :param batch_size: Runtime batch size used for searching. Defaults to
            :data:`DEFAULT_INDEX_BATCH_SIZE`.
        :param log_interval: Number of queried items between progress updates.
        :param display: Progress display mode.
        :param search_kwargs: Extra search options forwarded to raw search.
        :return: A pair ``(context_ids, scores)``.
        """

        def get_batch():
            batch = []
            for item in query:
                batch.append(item)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
            return

        scores = []
        indices = []
        total = len(query) if hasattr(query, "__len__") else None
        with SimpleProgressLogger(
            logger, total, interval=log_interval, display=display
        ) as p_logger:
            for q in get_batch():
                r = self.search(q, top_k, **search_kwargs)
                indices.extend(r[0])
                scores.append(r[1])
                p_logger.update(step=len(q), desc="Searching")
        return indices, np.concatenate(scores, axis=0)

    def search(
        self,
        query: list[Any],
        top_k: int,
        **search_kwargs,
    ) -> tuple[list[list[str]], np.ndarray]:
        """Search the context index.

        The underlying raw index may return multiple rows for the same context.
        Scores are merged according to ``self.cfg.merge_method`` before the
        final top-k context IDs are returned.

        :param query: Query items accepted by the underlying raw index.
        :param top_k: Number of context-level hits to return per query.
        :param search_kwargs: Extra search options forwarded to raw search.
        :return: A pair ``(context_ids, scores)``.
        """
        indices_batch, scores_batch = self.raw_index.search(
            query, top_k * self.max_field_num, **search_kwargs
        )

        new_indices = []
        new_scores = []
        for indices, scores in zip(indices_batch, scores_batch):
            retrieved = defaultdict(list)
            for idx, score in zip(indices, scores):
                context_id = self.index_to_context_id[idx]
                retrieved[context_id].append(score)

            for context_id in retrieved:
                match self.cfg.merge_method:
                    case "max":
                        retrieved[context_id] = max(retrieved[context_id])
                    case "sum":
                        retrieved[context_id] = sum(retrieved[context_id])
                    case "concat":
                        retrieved[context_id] = retrieved[context_id][0]
                    case "mean":
                        retrieved[context_id] = sum(retrieved[context_id]) / len(
                            retrieved[context_id]
                        )
                    case _:
                        raise ValueError(
                            f"Unknown merge method: {self.cfg.merge_method}"
                        )

            sorted_indices = sorted(retrieved.items(), key=lambda x: x[1], reverse=True)
            new_indices.append([x[0] for x in sorted_indices[:top_k]])
            new_scores.append([x[1] for x in sorted_indices[:top_k]])

        return new_indices, np.array(new_scores)

    def insert_batch(
        self,
        context_ids: Iterable[str],
        data: Iterable[dict[str, Any]],
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        """Insert contexts in batches.

        This method is available only when the underlying raw index is addable.
        It updates the row/context mapping and optionally persists the complete
        context index.

        :param context_ids: Context IDs corresponding to ``data``.
        :param data: Context dictionaries to project and insert.
        :param batch_size: Runtime batch size used for insertion. Defaults to
            :data:`DEFAULT_INDEX_BATCH_SIZE`.
        :param log_interval: Number of inserted raw rows between progress
            updates.
        :param display: Progress display mode.
        :return: None.
        """
        assert self.raw_index.is_addable, "Current index is not addable."
        row_context_ids = []
        offset = len(self.raw_index)

        def get_data_batch() -> Generator[list[Any], None, None]:
            batch = []
            for context_id, item in self._iter_index_data(context_ids, data):
                batch.append(item)
                row_context_ids.append(context_id)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
            return

        with SimpleProgressLogger(
            logger, interval=log_interval, display=display
        ) as p_logger:
            for batch in get_data_batch():
                self.raw_index.insert(batch)
                p_logger.update(step=len(batch), desc="Adding data")

        for idx, context_id in enumerate(row_context_ids):
            row_index = offset + idx
            self.context_id_to_index[context_id].append(row_index)
            self.index_to_context_id[row_index] = context_id

        return

    def insert(
        self,
        context_ids: list[str],
        data: list[dict[str, Any]],
    ) -> None:
        """Insert one batch of contexts.

        :param context_ids: Context IDs corresponding to ``data``.
        :param data: Context dictionaries to project and insert.
        :return: None.
        """
        assert len(context_ids) == len(data), (
            "The length of context_ids and data should be the same."
        )
        assert self.raw_index.is_addable, "Current index is not addable."
        offset = len(self.raw_index)
        rows = list(self._iter_index_data(context_ids, data))
        if len(rows) == 0:
            return

        row_context_ids = [row[0] for row in rows]
        self.raw_index.insert([row[1] for row in rows])
        for idx, context_id in enumerate(row_context_ids):
            row_index = offset + idx
            self.context_id_to_index[context_id].append(row_index)
            self.index_to_context_id[row_index] = context_id

        return

    def clear(self) -> None:
        """Clear in-memory state.

        :return: None.
        """
        self._reset_mapping()
        self.raw_index.clear()
        return

    def save_to_local(self, index_path: str) -> None:
        """Save the complete logical context index to disk.

        The context root stores the context config and row/context mapping. Raw
        artifacts are saved under the ``raw/`` child directory when the raw
        index is non-empty.

        :param index_path: Logical context-index root.
        :return: None.
        """
        os.makedirs(index_path, exist_ok=True)
        logger.info(f"Serializing context index to {index_path}")

        self.cfg.dump(os.path.join(index_path, "config.yaml"))
        with open(os.path.join(index_path, "cls.id"), "w", encoding="utf-8") as f:
            f.write(self.__class__.__name__)

        raw_path = os.path.join(index_path, "raw")
        if os.path.exists(raw_path):
            shutil.rmtree(raw_path)
        if len(self.raw_index) > 0:
            self.raw_index.save_to_local(raw_path)
        with open(os.path.join(index_path, "context_mapping.pkl"), "wb") as f:
            pickle.dump(
                {
                    "context_id_to_index": self.context_id_to_index,
                    "index_to_context_id": self.index_to_context_id,
                    "max_field_num": self.max_field_num,
                },
                f,
            )
        return

    @staticmethod
    def load_from_local(index_path: str, **kwargs) -> "ContextIndexBase":
        """Load a context index from a logical artifact directory.

        :param index_path: Directory containing context config, mapping, and
            the ``raw/`` child artifact directory.
        :param kwargs: Extra constructor dependencies forwarded to the raw
            index, such as dense encoders.
        :return: Loaded context-level index.
        """
        assert os.path.exists(index_path), f"Index path {index_path} does not exist."
        id_path = os.path.join(index_path, "cls.id")
        assert os.path.exists(id_path), f"Index ID file {id_path} does not exist."
        with open(id_path, "r", encoding="utf-8") as f:
            index_name = f.read().strip()

        index_cls = RETRIEVER_INDEX[index_name]["item"]
        config_cls = RETRIEVER_INDEX[index_name]["config_class"]
        config_path = os.path.join(index_path, "config.yaml")
        assert os.path.exists(config_path), (
            f"Configuration file {config_path} does not exist."
        )
        cfg = config_cls.load(config_path)
        index = index_cls(cfg, **kwargs)
        index._load_state_from_local(index_path, **kwargs)
        index._check_mapping_consistency()
        return index

    @property
    def is_addable(self) -> bool:
        """Whether the context index supports incremental insertion.

        :return: ``True`` when the underlying raw index is addable.
        """
        return self.raw_index.is_addable

    def __len__(self) -> int:
        """Return the number of indexed contexts.

        :return: Number of distinct context IDs in the mapping.
        """
        return len(self.context_id_to_index)

    @property
    def infimum(self) -> float:
        """Return the lower bound of context-level scores.

        :return: Minimum possible score inherited from the raw index.
        """
        return self.raw_index.infimum

    @property
    def supremum(self) -> float:
        """Return the upper bound of context-level scores.

        :return: Maximum possible score inherited from the raw index.
        """
        return self.raw_index.supremum


@configure
class DenseRawIndexBaseConfig:
    """Configuration for dense raw indexes.

    :param distance_function: Vector distance or similarity function. Available
        choices are ``"IP"``, ``"L2"``, and ``"COS"``. Defaults to ``"IP"``.
    """

    distance_function: Annotated[str, Choices("IP", "L2", "COS")] = "IP"


class DenseRawIndexBase(RawIndexBase):
    """Base class for dense row-level indexes.

    Dense raw indexes use injected encoders to convert query and passage data
    into vectors before indexing or searching.
    """

    config_cls: type[DenseRawIndexBaseConfig] = DenseRawIndexBaseConfig

    def __init__(
        self,
        cfg: DenseRawIndexBaseConfig,
        query_encoder: EncoderProtocol,
        passage_encoder: EncoderProtocol | None = None,
    ):
        """Create a dense raw index.

        :param cfg: Dense raw index configuration.
        :param query_encoder: Encoder used for search queries.
        :param passage_encoder: Optional encoder used for indexed passages. If
            omitted, ``query_encoder`` is reused.
        """
        super().__init__(cfg)
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder or query_encoder
        self.distance_function = self.cfg.distance_function
        return

    def encode_data_batch(
        self,
        data: Iterable[Any],
        is_query: bool = False,
        use_memmap: bool = True,
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        scratch_path: str | None = None,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        """Encode data into embeddings in batches.

        :param data: Query or passage data accepted by the selected encoder.
        :param is_query: Whether to encode with the query encoder. If
            ``False``, the passage encoder is used.
        :param use_memmap: Whether to stage embeddings through a temporary
            memory map to reduce peak memory use.
        :param batch_size: Runtime batch size used for encoder calls. Defaults
            to :data:`DEFAULT_INDEX_BATCH_SIZE`.
        :param scratch_path: Optional directory for temporary embedding files.
            If omitted, uses ``FLEXRAG_CACHE_DIR/embeddings``.
        :param log_interval: Number of encoded items between progress updates.
        :param display: Progress display mode.
        :return: Encoded embeddings.
        """
        if use_memmap:
            if scratch_path is None:
                mmap_path = os.path.join(FLEXRAG_CACHE_DIR, "embeddings")
            else:
                mmap_path = os.path.join(scratch_path, "embeddings")
            os.makedirs(mmap_path, exist_ok=True)
        else:
            mmap_path = None

        def get_batch() -> Generator[list[Any], None, None]:
            batch = []
            for item in data:
                batch.append(item)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
            return

        embeddings = []
        n_embeddings = 0
        with SimpleProgressLogger(
            logger, interval=log_interval, display=display
        ) as p_logger:
            for batch in get_batch():
                emb = self.encode_data(batch, is_query=is_query)
                if mmap_path is not None:
                    file_name = os.path.join(mmap_path, f"{uuid4()}.npy")
                    np.save(file_name, emb)
                    embeddings.append(file_name)
                else:
                    embeddings.append(emb)
                n_embeddings += emb.shape[0]
                p_logger.update(step=len(batch), desc="Encoding data")

        if isinstance(embeddings[0], str):
            logger.info("Copying embeddings to memory map")
            assert mmap_path is not None
            emb_path = embeddings[0]
            emb = np.load(emb_path)
            emb_map = np.memmap(
                os.path.join(mmap_path, "embeddings.npy"),
                dtype=np.float32,
                mode="w+",
                shape=(n_embeddings, emb.shape[1]),
            )
            idx = 0
            for emb_path in embeddings:
                emb = np.load(emb_path)
                emb_map[idx : idx + emb.shape[0]] = emb
                idx += emb.shape[0]
                del emb
                os.remove(emb_path)
            embeddings = emb_map
        else:
            embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

    def encode_data(self, data: list[Any], is_query: bool = False) -> np.ndarray:
        """Encode one batch of data and normalize it for the metric.

        :param data: Query or passage data accepted by the selected encoder.
        :param is_query: Whether to encode with the query encoder. If
            ``False``, the passage encoder is used.
        :return: Float32 embeddings ready for the raw index.
        """
        if is_query:
            encoder = self.query_encoder
        else:
            encoder = self.passage_encoder
        assert encoder is not None, "Encoder is not set."
        embeds = encoder.encode(data).astype("float32")
        return self._normalize_embeddings_for_metric(embeds)

    def _normalize_embeddings_for_metric(self, embeds: np.ndarray) -> np.ndarray:
        if self.distance_function != "COS" or embeds.size == 0:
            return embeds
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return embeds / norms

    def _postprocess_scores(self, scores: np.ndarray) -> np.ndarray:
        if self.distance_function == "L2":
            return -scores
        return scores

    def add_embeddings_batch(
        self,
        embeds: np.ndarray,
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        """Add embeddings to the raw index in batches.

        :param embeds: Embeddings to add to the raw index.
        :param batch_size: Runtime batch size used for adding embeddings.
            Defaults to :data:`DEFAULT_INDEX_BATCH_SIZE`.
        :param log_interval: Number of embeddings between progress updates.
        :param display: Progress display mode.
        :return: None.
        """
        with SimpleProgressLogger(
            logger, embeds.shape[0], log_interval, display=display
        ) as p_logger:
            for i in range(0, embeds.shape[0], batch_size):
                batch_embeds = embeds[i : i + batch_size]
                self.add_embeddings(batch_embeds)
                p_logger.update(step=batch_embeds.shape[0], desc="Adding embeddings")
        return

    @abstractmethod
    def add_embeddings(self, embeds: np.ndarray) -> None:
        """Add one batch of embeddings to the raw index.

        :param embeds: Embeddings to add.
        :return: None.
        """
        return

    def insert(self, data: list[Any]) -> None:
        """Encode and insert one batch of passage data.

        :param data: Passage data accepted by the passage encoder.
        :return: None.
        """
        embeddings = self.encode_data(data, is_query=False)
        self.add_embeddings(embeddings)
        return

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Return the embedding dimension used by this raw index.

        :return: Embedding dimension.
        """
        return

    @property
    def infimum(self) -> float:
        """Return the lower bound of dense similarity scores.

        :return: Minimum possible score for the configured distance function.
        """
        if self.distance_function == "L2":
            return float("-inf")
        if self.distance_function == "COS":
            return -1.0
        return float("-inf")

    @property
    def supremum(self) -> float:
        """Return the upper bound of dense similarity scores.

        :return: Maximum possible score for the configured distance function.
        """
        if self.distance_function == "L2":
            return 0.0
        if self.distance_function == "COS":
            return 1.0
        return float("inf")


RETRIEVER_INDEX = Register[ContextIndexBase]("index")
