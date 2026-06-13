import os
import pickle
import shutil
from collections import defaultdict
from typing import Annotated, Any, Generator, Iterable, Optional

import numpy as np

from flexrag.common import (
    __VERSION__,
    LOGGER_MANAGER,
    Choices,
    ProgressDisplay,
    SimpleProgressLogger,
    configure,
    trace,
)
from flexrag.common.configure import extract_config
from flexrag.common.database import (
    LMDBRetrieverDatabase,
    NaiveRetrieverDatabase,
    RetrieverDatabaseBase,
)
from flexrag.common.dataclasses import Context, RetrievedContext

from .index import RetrieverIndexBase
from .retriever_base import RETRIEVERS, LocalRetriever, LocalRetrieverConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retreviers.flex")


RETRIEVER_CARD_TEMPLATE = """---
language: en
library_name: FlexRAG
tags:
- FlexRAG
- retrieval
- search
- lexical
- RAG
---

# FlexRAG Retriever

This is a {retriever_type} created with the [`FlexRAG`](https://github.com/ictnlp/flexrag) library (version `{version}`).

## Installation

You can install the `FlexRAG` library with `pip`:

```bash
pip install flexrag
```

## Loading a `FlexRAG` retriever

You can use this retriever for information retrieval tasks. Here is an example:

```python
from flexrag.retriever import LocalRetriever

{load_example}

# You can retrieve now
results = retriever.search("Who is Bruce Wayne?")
```

FlexRAG Related Links:
* 📚[Documentation](https://flexrag.readthedocs.io/en/latest/)
* 💻[GitHub Repository](https://github.com/ictnlp/flexrag)
"""


def _build_retriever_card(
    retriever_type: str,
    version: str,
    repo_path: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> str:
    if repo_id is not None:
        load_example = (
            "# Load the retriever from the HuggingFace Hub\n"
            f'retriever = LocalRetriever.load_from_hub("{repo_id}")'
        )
    else:
        assert repo_path is not None, "repo_path must be provided when repo_id is None."
        load_example = (
            "# Load the retriever from a local path\n"
            f'retriever = LocalRetriever.load_from_local("{repo_path}")'
        )
    return RETRIEVER_CARD_TEMPLATE.format(
        retriever_type=retriever_type,
        version=version,
        load_example=load_example,
    )


@configure
class IndexFieldsConfig:
    """Configuration for binding a retriever index to context fields.

    :param indexed_fields: Fields to index. If ``None``, all fields are indexed.
        Defaults to None.
    :type indexed_fields: Optional[list[str]]
    :param merge_method: Method to merge scores from multiple indexed fields of
        the same context. Available choices are "max", "sum", "mean", and
        "concat". Defaults to "max".
    :type merge_method: str
    """

    indexed_fields: Optional[list[str]] = None
    merge_method: Annotated[str, Choices("max", "sum", "mean", "concat")] = "max"


class _IndexBinding:
    """Internal binding between a flat index and FlexRetriever context fields."""

    def __init__(
        self,
        index: RetrieverIndexBase,
        fields_config: IndexFieldsConfig | None = None,
    ):
        self.index = index
        self.fields_config = extract_config(
            fields_config or IndexFieldsConfig(),
            IndexFieldsConfig,
        )

        if self.index.cfg.index_path is not None:
            mapping_path = os.path.join(
                self.index.cfg.index_path, "context_mapping.pkl"
            )
            if os.path.exists(mapping_path):
                with open(mapping_path, "rb") as f:
                    mapping = pickle.load(f)
                self.context_id_to_index = mapping["context_id_to_index"]
                self.index_to_context_id = mapping["index_to_context_id"]
                self.max_field_num = mapping["max_field_num"]
            else:
                assert len(self.index) == 0, (
                    "The index should be empty before building the field binding."
                )
                self._reset_mapping()
        else:
            assert len(self.index) == 0, "The index should be empty before building."
            self._reset_mapping()

        assert len(self.index_to_context_id) == len(self.index), (
            "The length of the index and the context-id mapping should be the same."
        )
        return

    def _reset_mapping(self) -> None:
        self.index_to_context_id: dict[int, str] = {}
        self.context_id_to_index: dict[str, list[int]] = defaultdict(list)
        self.max_field_num = 1
        return

    def _iter_index_data(
        self,
        context_ids: Iterable[str],
        data: Iterable[dict[str, Any]],
    ) -> Generator[tuple[str, Any], None, None]:
        for context_id, item in zip(context_ids, data):
            if self.fields_config.indexed_fields is None:
                indexed_fields = list(item.keys())
            else:
                indexed_fields = [
                    field for field in self.fields_config.indexed_fields if field in item
                ]

            if self.fields_config.merge_method == "concat":
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
        index_path: Optional[str] = None,
    ) -> None:
        self._reset_mapping()
        row_context_ids: list[str] = []

        def get_data() -> Generator[Any, None, None]:
            for context_id, item in self._iter_index_data(context_ids, data):
                row_context_ids.append(context_id)
                yield item
            return

        self.index.build_index(get_data())
        for idx, context_id in enumerate(row_context_ids):
            self.context_id_to_index[context_id].append(idx)
            self.index_to_context_id[idx] = context_id

        index_path = index_path or self.index.cfg.index_path
        if index_path is not None:
            self.save_to_local(index_path=index_path)
        return

    def search_batch(
        self,
        query: list[Any],
        top_k: int,
        batch_size: int | None = None,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
        **search_kwargs,
    ) -> tuple[list[list[str]], np.ndarray]:
        batch_size = batch_size or self.index.cfg.batch_size

        def get_batch():
            batch = []
            for item in query:
                batch.append(item)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

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
        indices_batch, scores_batch = self.index.search(
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
                match self.fields_config.merge_method:
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
                            f"Unknown merge method: {self.fields_config.merge_method}"
                        )

            sorted_indices = sorted(retrieved.items(), key=lambda x: x[1], reverse=True)
            new_indices.append([x[0] for x in sorted_indices[:top_k]])
            new_scores.append([x[1] for x in sorted_indices[:top_k]])

        return new_indices, np.array(new_scores)

    def insert_batch(
        self,
        context_ids: Iterable[str],
        data: Iterable[dict[str, Any]],
        batch_size: Optional[int] = None,
        serialize: bool = True,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        assert self.index.is_addable, "Current index is not addable."
        batch_size = batch_size or self.index.cfg.batch_size
        row_context_ids = []
        offset = len(self.index)

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
                self.index.insert(batch)
                p_logger.update(step=len(batch), desc="Adding data")

        for idx, context_id in enumerate(row_context_ids):
            row_index = offset + idx
            self.context_id_to_index[context_id].append(row_index)
            self.index_to_context_id[row_index] = context_id

        if (self.index.cfg.index_path is not None) and serialize:
            self.save_to_local()
        return

    def insert(
        self,
        context_ids: list[str],
        data: list[dict[str, Any]],
        serialize: bool = True,
    ) -> None:
        assert len(context_ids) == len(data), (
            "The length of context_ids and data should be the same."
        )
        assert self.index.is_addable, "Current index is not addable."
        offset = len(self.index)
        rows = list(self._iter_index_data(context_ids, data))
        if len(rows) == 0:
            return

        row_context_ids = [row[0] for row in rows]
        self.index.insert([row[1] for row in rows])
        for idx, context_id in enumerate(row_context_ids):
            row_index = offset + idx
            self.context_id_to_index[context_id].append(row_index)
            self.index_to_context_id[row_index] = context_id

        if (self.index.cfg.index_path is not None) and serialize:
            self.save_to_local()
        return

    def clear(self) -> None:
        self._reset_mapping()
        self.index.clear()
        return

    def save_to_local(self, index_path: Optional[str] = None) -> None:
        index_path = index_path or self.index.cfg.index_path
        if index_path is None:
            raise ValueError("index_path is not set.")

        self.index.save_to_local(index_path)
        config_path = os.path.join(index_path, "index_fields_config.yaml")
        self.fields_config.dump(config_path)

        context_mapping_path = os.path.join(index_path, "context_mapping.pkl")
        with open(context_mapping_path, "wb") as f:
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
    def load_from_local(index_path: str, **kwargs) -> "_IndexBinding":
        index = RetrieverIndexBase.load_from_local(index_path, **kwargs)
        config_path = os.path.join(index_path, "index_fields_config.yaml")
        assert os.path.exists(config_path), (
            f"Configuration file not found in {index_path}."
        )
        fields_config = IndexFieldsConfig.load(config_path)
        return _IndexBinding(index=index, fields_config=fields_config)

    @property
    def is_addable(self) -> bool:
        return self.index.is_addable

    def __len__(self) -> int:
        return len(self.context_id_to_index)

    @property
    def infimum(self) -> float:
        return self.index.infimum

    @property
    def supremum(self) -> float:
        return self.index.supremum


@configure
class FlexRetrieverConfig(LocalRetrieverConfig):
    """Configuration class for FlexRetriever."""


@RETRIEVERS("flex", config_class=FlexRetrieverConfig)
class FlexRetriever(LocalRetriever):
    """FlexRetriever is a retriever implemented by FlexRAG team.
    FlexRetriever supports multi-index and multi-field retrieval.
    """

    cfg: FlexRetrieverConfig

    def __init__(self, cfg: FlexRetrieverConfig) -> None:
        super().__init__(cfg)
        self.cfg = extract_config(cfg, FlexRetrieverConfig)
        # load the retriever if the retriever_path is set
        self.database = self._load_database()
        self.index_table: dict[str, _IndexBinding] = self._load_index()

        # consistency check
        self._check_consistency()
        return

    @trace("retriever.flex_retriever.add_passages")
    def add_passages(
        self,
        passages: Iterable[Context],
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ):

        def get_batch() -> Generator[tuple[list[dict], list[str]], None, None]:
            batch = []
            ids = []
            for passage in passages:
                if len(batch) == self.cfg.batch_size:
                    yield batch, ids
                    batch = []
                    ids = []
                data = passage.data.copy()
                ids.append(passage.context_id)
                batch.append(data)
            if batch:
                yield batch, ids
            return

        # add data to database
        context_ids = []
        with SimpleProgressLogger(
            logger, interval=log_interval, display=display
        ) as p_logger:
            for batch, ids in get_batch():
                self.database[ids] = batch
                context_ids.extend(ids)
                p_logger.update(step=len(batch), desc="Adding passages")

        # update the indexes
        self._update_index(
            context_ids,
            log_interval=log_interval,
            display=display,
        )
        logger.info("Finished adding passages.")
        return

    @trace("retriever.flex_retriever.search")
    def _search(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        top_k = search_kwargs.pop("top_k", self.cfg.top_k)
        used_indexes = search_kwargs.pop("used_indexes", None)
        merge_method = search_kwargs.pop("indexes_merge_method", "rrf")
        merge_weights = search_kwargs.pop("indexes_merge_weights", None)
        rrf_base = search_kwargs.pop("rrf_base", 60)
        if used_indexes is None:
            used_indexes = list(self.index_table.keys())
        for index_name in used_indexes:
            assert index_name in self.index_table, f"Index {index_name} not found."
        assert len(used_indexes) > 0, "`used_indexes` is empty."

        # retrieve indices using `used_indexes`
        all_context_ids = []
        all_scores = []
        for index_name in used_indexes:
            r = self.index_table[index_name].search(query, top_k, **search_kwargs)
            all_context_ids.append(r[0])
            all_scores.append(r[1])

        # merge the indices and scores
        merged_ids: list[list[str]] = []
        merged_scores: list[list[float]] = []
        if len(all_scores) == 1:  # only one index is activated
            merged_scores = all_scores[0].tolist()
            merged_ids = all_context_ids[0]
        else:  # merge multiple indexes
            match merge_method:
                case "rrf":
                    # prepare merge weights
                    if merge_weights is not None:
                        assert len(merge_weights) == len(used_indexes)
                        merge_weights = [
                            i / sum(merge_weights)
                            for i in merge_weights
                        ]
                    else:
                        merge_weights = [1.0 / len(all_scores)] * len(all_scores)
                    # recompute the scores according to the rank
                    for i in range(len(query)):
                        scores_dict = defaultdict(float)
                        for ctx_ids, scores, merge_weight in zip(
                            all_context_ids, all_scores, merge_weights
                        ):
                            sort_ranks = scores[i].argsort()[::-1] + 1
                            for ctx_id, rank in zip(ctx_ids[i], sort_ranks):
                                scores_dict[ctx_id] += merge_weight / (
                                    rank + rrf_base
                                )
                        sorted_items = sorted(scores_dict.items(), key=lambda x: -x[1])
                        merged_ids.append([item[0] for item in sorted_items][:top_k])
                        merged_scores.append([item[1] for item in sorted_items][:top_k])
                case "linear":
                    # prepare merge weights
                    if merge_weights is not None:
                        assert len(merge_weights) == len(used_indexes)
                        merge_weights = [
                            i / sum(merge_weights)
                            for i in merge_weights
                        ]
                    else:
                        merge_weights = [1.0 / len(all_scores)] * len(all_scores)
                    # According to "An Analysis of Fusion Functions for Hybrid Retrieval",
                    # we employ the TMM normalization method to normalize the scores.
                    if any(
                        self.index_table[index_name].infimum == float("-inf")
                        for index_name in used_indexes
                    ):
                        use_infimum = False
                    else:
                        use_infimum = True
                    for n in range(len(used_indexes)):
                        index_name = used_indexes[n]
                        if use_infimum:
                            infimum = self.index_table[index_name].infimum
                        else:
                            infimum = all_scores[n].min(axis=1, keepdims=True)
                        all_scores[n] = (all_scores[n] - infimum) / (
                            all_scores[n].max(axis=1, keepdims=True) - infimum
                        )
                    # merge the scores
                    for i in range(len(query)):
                        scores_dict = defaultdict(float)
                        for ctx_ids, scores, merge_weight in zip(
                            all_context_ids, all_scores, merge_weights
                        ):
                            for ctx_id, score in zip(ctx_ids[i], scores[i]):
                                scores_dict[ctx_id] += score * merge_weight
                        sorted_items = sorted(scores_dict.items(), key=lambda x: -x[1])
                        merged_ids.append([item[0] for item in sorted_items][:top_k])
                        merged_scores.append([item[1] for item in sorted_items][:top_k])
                case _:
                    raise ValueError(f"Unknown merge method: {merge_method}")

        # form the final results
        results: list[list[RetrievedContext]] = []
        for i, (q, score, context_id) in enumerate(
            zip(query, merged_scores, merged_ids)
        ):
            results.append([])
            ctxs = self.database[context_id]
            for j, (s, ctx_id, ctx) in enumerate(zip(score, context_id, ctxs)):
                results[-1].append(
                    RetrievedContext(
                        context_id=ctx_id,
                        retriever="FlexRetriever",
                        query=q,
                        score=float(s),
                        data=ctx,
                    )
                )
        return results

    def clear(self) -> None:
        # clear the indexes
        for index_name in self.index_table:
            self.index_table[index_name].clear()

        # clear the database
        self.database.clear()

        # clear the directory
        if self.cfg.retriever_path is not None:
            if os.path.exists(self.cfg.retriever_path):
                shutil.rmtree(self.cfg.retriever_path)
        return

    def __len__(self) -> int:
        return len(self.database)

    @property
    def fields(self) -> list[str]:
        return self.database.fields

    @trace("retriever.flex_retriever.add_index")
    def add_index(
        self,
        index_name: str,
        index: RetrieverIndexBase,
        fields_config: IndexFieldsConfig | None = None,
    ) -> None:
        """Add an index to the retriever.

        :param index_name: Name of the index.
        :type index_name: str
        :param index: Prepared flat index to bind to retriever fields.
        :type index: RetrieverIndexBase
        :param fields_config: Field binding configuration. If None, all fields
            are indexed and field hits are merged with ``max``.
        :type fields_config: Optional[IndexFieldsConfig]
        :raises ValueError: If the index name already exists.
        :return: None
        :rtype: None
        """
        # check if the index name is valid
        if index_name in self.index_table:
            raise ValueError(
                f"Index {index_name} already exists. Please remove it first."
            )

        # prepare index path
        if self.cfg.retriever_path is not None:
            index_path = os.path.join(self.cfg.retriever_path, "indexes", index_name)
        else:
            index_path = None

        binding = _IndexBinding(index=index, fields_config=fields_config)
        if len(self.database) > 0:
            binding.build_index(self.database.ids, self.database.values(), index_path)

        # add index to the index table
        self.index_table[index_name] = binding
        self._check_consistency()
        logger.info(f"Finished adding index: {index_name}")
        return

    def remove_index(self, index_name: str) -> None:
        """Remove an index from the retriever.

        :param index_name: Name of the index.
        :type index_name: str
        :raises ValueError: If the index name does not exist.
        :return: None
        :rtype: None
        """
        if index_name not in self.index_table:
            raise ValueError(f"Index {index_name} does not exist.")

        # remove the index
        index = self.index_table.pop(index_name)
        index.clear()

        # update the configuration
        return

    def save_to_local(self, retriever_path: str = None) -> None:
        # check if the retriever is serializable
        if self.cfg.retriever_path is not None:
            if retriever_path == self.cfg.retriever_path:
                return  # skip saving if the path is the same
        else:
            assert retriever_path is not None, "`retriever_path` is not set."
            self.cfg.retriever_path = retriever_path
        self._check_retriever_path(retriever_path)
        logger.info(f"Serializing retriever to {retriever_path}")

        # save the database
        def get_data() -> Generator[tuple[list[str], list[dict]], None, None]:
            batch_ids = []
            batch_data = []
            for ctx_id, ctx in self.database.items():
                # unify the schema
                # FIXME: if the schema is not consistent, we need to handle it
                ctx = {k: ctx.get(k, "") for k in self.fields}
                batch_ids.append(ctx_id)
                batch_data.append(ctx)
                if len(batch_ids) == self.cfg.batch_size:
                    yield batch_ids, batch_data
                    batch_ids = []
                    batch_data = []
            if batch_ids:
                yield batch_ids, batch_data
            return

        new_db = LMDBRetrieverDatabase(os.path.join(retriever_path, "database.lmdb"))
        for batch_ids, batch_data in get_data():
            new_db[batch_ids] = batch_data
        self.database = new_db

        # save the index
        for index_name, index in self.index_table.items():
            index_path = os.path.join(retriever_path, "indexes", index_name)
            index.save_to_local(index_path)
        return

    def detach(self):
        """Detach the retriever from the local disk to memory.
        This function will not delete the database or the indexes."""

        def get_data() -> Generator[tuple[list[str], list[dict]], None, None]:
            batch_ids = []
            for ctx_id in self.database.ids:
                batch_ids.append(ctx_id)
                if len(batch_ids) == self.cfg.batch_size:
                    yield batch_ids, self.database[batch_ids]
                    batch_ids = []
            if batch_ids:
                yield batch_ids, self.database[batch_ids]
            return

        # detach the database
        if isinstance(self.database, LMDBRetrieverDatabase):
            new_db = NaiveRetrieverDatabase()
            for batch_ids, batch_data in get_data():
                new_db[batch_ids] = batch_data
            self.database = new_db

        # detach the indexes
        for index_name, index in self.index_table.items():
            index.index.cfg.index_path = None

        # update the configuration
        self.cfg.retriever_path = None
        return

    def _update_index(
        self,
        context_ids: list[str],
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        def get_data() -> Generator[tuple[Any, int], None, None]:
            for ctx_id in context_ids:
                yield self.database[ctx_id]

        # update index
        for index_name, index in self.index_table.items():
            # prepare index path
            if self.cfg.retriever_path is not None:
                index_path = os.path.join(
                    self.cfg.retriever_path, "indexes", index_name
                )
            else:
                index_path = None
            if index.is_addable:
                index.insert_batch(
                    context_ids,
                    get_data(),
                    serialize=True,
                    log_interval=log_interval,
                    display=display,
                )
            else:
                logger.warning(
                    f"Index {index_name} is not addable. Rebuilding the index."
                )
                index.clear()
                index.build_index(context_ids, get_data(), index_path)
        return

    def _load_database(self) -> RetrieverDatabaseBase:
        if self.cfg.retriever_path is not None:
            database_path = os.path.join(self.cfg.retriever_path, "database.lmdb")
            database = LMDBRetrieverDatabase(database_path)
        else:
            database = NaiveRetrieverDatabase()
        return database

    def _load_index(self) -> dict[str, _IndexBinding]:
        # load indexes
        indexes = {}
        if self.cfg.retriever_path is None:
            return indexes
        if not os.path.exists(os.path.join(self.cfg.retriever_path, "indexes")):
            return indexes
        indexes_names = os.listdir(os.path.join(self.cfg.retriever_path, "indexes"))
        for index_name in indexes_names:
            index_path = os.path.join(self.cfg.retriever_path, "indexes", index_name)
            index = _IndexBinding.load_from_local(index_path)
            indexes[index_name] = index
        return indexes

    def _check_consistency(self) -> None:
        if self.cfg.retriever_path is not None:
            self._check_retriever_path(self.cfg.retriever_path)
        for index_name, index in self.index_table.items():
            assert len(index) == len(self.database), "Index and database size mismatch"
        return

    def _check_retriever_path(self, retriever_path: str) -> None:
        if not os.path.exists(retriever_path):
            os.makedirs(retriever_path)

        # save the retriever card
        card_path = os.path.join(retriever_path, "README.md")
        if not os.path.exists(card_path):
            retriever_card = _build_retriever_card(
                retriever_type=self.__class__.__name__,
                version=__VERSION__,
                repo_path=self.cfg.retriever_path,
            )
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(retriever_card)

        # save the configuration
        cfg_path = os.path.join(retriever_path, "config.yaml")
        if not os.path.exists(cfg_path):
            self.cfg.dump(cfg_path)
        id_path = os.path.join(retriever_path, "cls.id")
        if not os.path.exists(id_path):
            with open(id_path, "w", encoding="utf-8") as f:
                f.write(self.__class__.__name__)

    def __getitem__(self, context_id: str) -> dict:
        return self.database[context_id]
