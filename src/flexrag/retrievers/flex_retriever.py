import os
import shutil
from collections import defaultdict
from typing import Any, Generator, Iterable, Optional

from flexrag.common import (
    __VERSION__,
    LOGGER_MANAGER,
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

from .index import ContextIndexBase
from .retriever_base import (
    DEFAULT_TOP_K,
    RETRIEVERS,
    LocalRetriever,
    LocalRetrieverConfig,
)

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
class FlexRetrieverConfig(LocalRetrieverConfig):
    """Configuration for FlexRetriever.

    :param batch_size: Number of contexts processed per batch when adding
        passages, updating indexes, serializing the database, or running search.
        Defaults to 32.
    :param query_preprocess_pipeline: Text processing pipeline applied to
        queries before search unless ``no_preprocess=True`` is passed at
        runtime.
    """


@RETRIEVERS("flex", config_class=FlexRetrieverConfig)
class FlexRetriever(LocalRetriever):
    """FlexRetriever is a retriever implemented by FlexRAG team.
    FlexRetriever supports multi-index and multi-field retrieval.
    """

    cfg: FlexRetrieverConfig

    def __init__(
        self,
        cfg: FlexRetrieverConfig,
        retriever_path: Optional[str] = None,
    ) -> None:
        super().__init__(cfg)
        self.cfg = extract_config(cfg, FlexRetrieverConfig)
        self._retriever_path = retriever_path
        self.database = self._load_database()
        self.index_table: dict[str, ContextIndexBase] = self._load_index()

        # consistency check
        self._check_consistency()
        return

    @property
    def retriever_path(self) -> Optional[str]:
        """Return the local artifact root attached to this retriever.

        :return: The local retriever artifact path, or ``None`` when the
            retriever is detached from disk.
        """
        return self._retriever_path

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
        if self.retriever_path is not None:
            self._save_metadata(self.retriever_path)
        logger.info("Finished adding passages.")
        return

    @trace("retriever.flex_retriever.search")
    def _search(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        top_k = search_kwargs.pop("top_k", DEFAULT_TOP_K)
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

        if self.retriever_path is not None:
            self._save_metadata(self.retriever_path)
            for index_name, index in self.index_table.items():
                index.save_to_local(self._get_index_path(index_name))
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
        index: ContextIndexBase,
    ) -> None:
        """Add an index to the retriever.

        :param index_name: Name of the index.
        :param index: Prepared context-level index.
        :raises ValueError: If the index name already exists.
        :return: None
        """
        # check if the index name is valid
        if index_name in self.index_table:
            raise ValueError(
                f"Index {index_name} already exists. Please remove it first."
            )

        # prepare index path
        if self.retriever_path is not None:
            index_path = self._get_index_path(index_name)
        else:
            index_path = None

        if len(self.database) > 0:
            scratch_path = (
                os.path.join(index_path, "raw") if index_path is not None else None
            )
            index.build_index(
                self.database.ids,
                self.database.values(),
                scratch_path=scratch_path,
            )
            if index_path is not None:
                index.save_to_local(index_path)
        elif index_path is not None:
            index.save_to_local(index_path)

        # add index to the index table
        self.index_table[index_name] = index
        if self.retriever_path is not None:
            self._save_metadata(self.retriever_path)
        self._check_consistency()
        logger.info(f"Finished adding index: {index_name}")
        return

    def remove_index(self, index_name: str) -> None:
        """Remove an index from the retriever.

        :param index_name: Name of the index.
        :raises ValueError: If the index name does not exist.
        :return: None
        """
        if index_name not in self.index_table:
            raise ValueError(f"Index {index_name} does not exist.")

        # remove the index
        index = self.index_table.pop(index_name)
        index.clear()
        if self.retriever_path is not None:
            index_path = self._get_index_path(index_name)
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            self._save_metadata(self.retriever_path)

        # update the configuration
        return

    def save_to_local(self, retriever_path: Optional[str] = None) -> None:
        # check if the retriever is serializable
        if retriever_path is None:
            retriever_path = self.retriever_path
        assert retriever_path is not None, "`retriever_path` is not set."
        retriever_path = os.fspath(retriever_path)
        self._retriever_path = retriever_path
        self._save_metadata(retriever_path)
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

        database_path = os.path.join(retriever_path, "database.lmdb")
        current_database_path = (
            self.database.database_path
            if isinstance(self.database, LMDBRetrieverDatabase)
            else None
        )
        if current_database_path is None or os.path.abspath(
            current_database_path
        ) != os.path.abspath(database_path):
            if os.path.exists(database_path):
                shutil.rmtree(database_path)
            new_db = LMDBRetrieverDatabase(database_path)
            for batch_ids, batch_data in get_data():
                new_db[batch_ids] = batch_data
            if isinstance(self.database, LMDBRetrieverDatabase):
                self.database.close()
            self.database = new_db

        # save the index
        for index_name, index in self.index_table.items():
            index.save_to_local(self._get_index_path(index_name))
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
            old_db = self.database
            new_db = NaiveRetrieverDatabase()
            for batch_ids, batch_data in get_data():
                new_db[batch_ids] = batch_data
            self.database = new_db
            old_db.close()

        # update the runtime state
        self._retriever_path = None
        return

    def _update_index(
        self,
        context_ids: list[str],
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        def get_data(ctx_ids: Iterable[str]) -> Generator[dict[str, Any], None, None]:
            for ctx_id in ctx_ids:
                yield self.database[ctx_id]

        # update index
        for index_name, index in self.index_table.items():
            # prepare index path
            if self.retriever_path is not None:
                index_path = self._get_index_path(index_name)
            else:
                index_path = None
            if index.is_addable:
                index.insert_batch(
                    context_ids,
                    get_data(context_ids),
                    log_interval=log_interval,
                    display=display,
                )
                if index_path is not None:
                    index.save_to_local(index_path)
            else:
                logger.warning(
                    f"Index {index_name} is not addable. Rebuilding the index."
                )
                index.clear()
                scratch_path = (
                    os.path.join(index_path, "raw") if index_path is not None else None
                )
                index.build_index(
                    self.database.ids,
                    self.database.values(),
                    scratch_path=scratch_path,
                )
                if index_path is not None:
                    index.save_to_local(index_path)
        return

    def _load_database(self) -> RetrieverDatabaseBase:
        if self.retriever_path is not None:
            database_path = os.path.join(self.retriever_path, "database.lmdb")
            database = LMDBRetrieverDatabase(database_path)
        else:
            database = NaiveRetrieverDatabase()
        return database

    def _load_index(self) -> dict[str, ContextIndexBase]:
        # load indexes
        indexes = {}
        if self.retriever_path is None:
            return indexes
        index_root = os.path.join(self.retriever_path, "indexes")
        if not os.path.exists(index_root):
            return indexes
        indexes_names = os.listdir(index_root)
        for index_name in indexes_names:
            index_path = self._get_index_path(index_name)
            index = ContextIndexBase.load_from_local(index_path)
            indexes[index_name] = index
        return indexes

    def _check_consistency(self) -> None:
        for index_name, index in self.index_table.items():
            assert len(index) == len(self.database), "Index and database size mismatch"
        return

    def _get_index_path(self, index_name: str) -> str:
        assert self.retriever_path is not None, "`retriever_path` is not set."
        return os.path.join(self.retriever_path, "indexes", index_name)

    def _save_metadata(self, retriever_path: str) -> None:
        os.makedirs(retriever_path, exist_ok=True)

        # save the retriever card
        card_path = os.path.join(retriever_path, "README.md")
        if not os.path.exists(card_path):
            retriever_card = _build_retriever_card(
                retriever_type=self.__class__.__name__,
                version=__VERSION__,
                repo_path=retriever_path,
            )
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(retriever_card)

        # save the configuration
        cfg_path = os.path.join(retriever_path, "config.yaml")
        self.cfg.dump(cfg_path)
        id_path = os.path.join(retriever_path, "cls.id")
        with open(id_path, "w", encoding="utf-8") as f:
            f.write(self.__class__.__name__)

    def __getitem__(self, context_id: str) -> dict:
        return self.database[context_id]
