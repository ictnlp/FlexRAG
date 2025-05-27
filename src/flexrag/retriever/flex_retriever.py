import os
import shutil
from collections import defaultdict
from typing import Annotated, Any, Generator, Iterable, Optional

from jinja2 import Template
from omegaconf import OmegaConf

from flexrag.database import (
    LMDBRetrieverDatabase,
    NaiveRetrieverDatabase,
    RetrieverDatabaseBase,
)
from flexrag.utils import (
    __VERSION__,
    LOGGER_MANAGER,
    TIME_METER,
    Choices,
    SimpleProgressLogger,
    configure,
)
from flexrag.utils.configure import extract_config
from flexrag.utils.dataclasses import Context, RetrievedContext

from .index import (
    RETRIEVER_INDEX,
    MultiFieldIndex,
    MultiFieldIndexConfig,
    RetrieverIndexConfig,
)
from .retriever_base import RETRIEVERS, LocalRetriever, LocalRetrieverConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retreviers.flex")


RETRIEVER_CARD_TEMPLATE = Template(
    """---
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

This is a {{ retriever_type }} created with the [`FlexRAG`](https://github.com/ictnlp/flexrag) library (version `{version}`).

## Installation

You can install the `FlexRAG` library with `pip`:

```bash
pip install flexrag
```

## Loading a `FlexRAG` retriever

You can use this retriever for information retrieval tasks. Here is an example:

```python
from flexrag.retriever import LocalRetriever

{% if repo_id is not none %}
# Load the retriever from the HuggingFace Hub
retriever = LocalRetriever.load_from_hub("{{ repo_id }}")
{% else %}
# Load the retriever from a local path
retriever = LocalRetriever.load_from_local("{{ repo_path }}")
{% endif %}

# You can retrieve now
results = retriever.search("Who is Bruce Wayne?")
```

FlexRAG Related Links:
* 📚[Documentation](https://flexrag.readthedocs.io/en/latest/)
* 💻[GitHub Repository](https://github.com/ictnlp/flexrag)
"""
)


@configure
class FlexRetrieverConfig(LocalRetrieverConfig):
    """Configuration class for FlexRetriever.

    :param indexes_merge_method: Method to merge the scores of multiple indexes.
        Available choices are "rrf" and "linear". Default is "rrf".
        * "rrf": Reciprocal Rank Fusion (RRF) method.
        * "linear": Linear combination of the scores.
    :type indexes_merge_method: str
    :param merge_weights: List of weights for each index. Default is None.
        If None, all indexes will be treated equally.
        This option is used in both "rrf" and "linear" methods.
    :type merge_weights: Optional[list[float]]
    :param used_indexes: List of indexes to use for retrieval. Default is None.
        If None, all indexes will be used.
    :type used_indexes: Optional[list[str]]
    :param rrf_base: Base for the RRF method. Default is 60.
        This option is only used when `indexes_merge_method` is "rrf".
    :type rrf_base: int
    """

    indexes_merge_method: Annotated[str, Choices("rrf", "linear")] = "rrf"
    indexes_merge_weights: Optional[list[float]] = None
    used_indexes: Optional[list[str]] = None
    rrf_base: int = 60


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
        self.index_table = self._load_index()

        # consistency check
        self._check_consistency()
        return

    @TIME_METER("flex_retriever", "add-passages")
    def add_passages(self, passages: Iterable[Context]):

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
        p_logger = SimpleProgressLogger(logger, interval=self.cfg.log_interval)
        for batch, ids in get_batch():
            self.database[ids] = batch
            context_ids.extend(ids)
            p_logger.update(step=len(batch), desc="Adding passages")

        # update the indexes
        self._update_index(context_ids)
        logger.info("Finished adding passages.")
        return

    @TIME_METER("flex_retriever", "search")
    def search(
        self,
        query: list[str] | str,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        if isinstance(query, str):
            query = [query]
        top_k = search_kwargs.pop("top_k", self.cfg.top_k)
        used_indexes = search_kwargs.pop("used_indexes", self.cfg.used_indexes)
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
            merge_method = search_kwargs.pop(
                "indexes_merge_method", self.cfg.indexes_merge_method
            )
            match merge_method:
                case "rrf":
                    # prepare merge weights
                    if self.cfg.indexes_merge_weights is not None:
                        assert len(self.cfg.indexes_merge_weights) == len(used_indexes)
                        merge_weights = [
                            i / sum(self.cfg.indexes_merge_weights)
                            for i in self.cfg.indexes_merge_weights
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
                                    rank + self.cfg.rrf_base
                                )
                        sorted_items = sorted(scores_dict.items(), key=lambda x: -x[1])
                        merged_ids.append([item[0] for item in sorted_items][:top_k])
                        merged_scores.append([item[1] for item in sorted_items][:top_k])
                case "linear":
                    # prepare merge weights
                    if self.cfg.indexes_merge_weights is not None:
                        assert len(self.cfg.indexes_merge_weights) == len(used_indexes)
                        merge_weights = self.cfg.indexes_merge_weights / sum(
                            self.cfg.indexes_merge_weights
                        )
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

    @TIME_METER("flex_retriever", "add-index")
    def add_index(
        self,
        index_name: str,
        index_config: RetrieverIndexConfig,  # type: ignore
        indexed_fields_config: MultiFieldIndexConfig,
    ) -> None:
        """Add an index to the retriever.

        :param index_name: Name of the index.
        :type index_name: str
        :param index_config: Configuration of the index.
        :type index_config: RetrieverIndexConfig
        :param indexed_fields_config: Configuration of the indexed fields.
        :type indexed_fields_config: MultiFieldIndexConfig
        :raises ValueError: If the index name already exists.
        :return: None
        :rtype: None
        """
        # check if the index name is valid
        if index_name in self.index_table:
            raise ValueError(
                f"Index {index_name} already exists. Please remove it first."
            )

        # prepare the index
        index = RETRIEVER_INDEX.load(index_config)
        index = MultiFieldIndex(indexed_fields_config, index)
        index.build_index(self.database.ids, self.database.values())

        # prepare index path
        if self.cfg.retriever_path is not None:
            index_path = os.path.join(self.cfg.retriever_path, "indexes", index_name)
        else:
            index_path = None
        if index_path is not None:
            index.save_to_local(index_path)

        # add index to the index table
        self.index_table[index_name] = index
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
        if index_name in self.cfg.used_indexes:
            self.cfg.used_indexes.remove(index_name)
        return

    def save_to_local(self, retriever_path: str = None) -> None:
        # check if the retriever is serializable
        if self.cfg.retriever_path is not None:
            if retriever_path == self.cfg.retriever_path:
                return  # skip saving if the path is the same
        else:
            assert retriever_path is not None, "`retriever_path` is not set."
            self.cfg.retriever_path = retriever_path
        if not os.path.exists(retriever_path):
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

    def _update_index(self, context_ids: list[str]) -> None:
        def get_data() -> Generator[tuple[Any, int], None, None]:
            for ctx_id in context_ids:
                yield self.database[ctx_id]

        for index_name, index in self.index_table.items():
            if index.is_addable:
                index.insert_batch(context_ids, get_data(), serialize=True)
            else:
                logger.warning(
                    f"Index {index_name} is not addable. Rebuilding the index."
                )
                index.clear()
                index.build_index(get_data())
        return

    def _load_database(self) -> RetrieverDatabaseBase:
        if self.cfg.retriever_path is not None:
            database_path = os.path.join(self.cfg.retriever_path, "database.lmdb")
            database = LMDBRetrieverDatabase(database_path)
        else:
            database = NaiveRetrieverDatabase()
        return database

    def _load_index(self) -> dict[str, MultiFieldIndex]:
        # load indexes
        indexes = {}
        if self.cfg.retriever_path is None:
            return indexes
        if not os.path.exists(os.path.join(self.cfg.retriever_path, "indexes")):
            return indexes
        indexes_names = os.listdir(os.path.join(self.cfg.retriever_path, "indexes"))
        for index_name in indexes_names:
            index_path = os.path.join(self.cfg.retriever_path, "indexes", index_name)
            index = MultiFieldIndex.load_from_local(index_path)
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
            retriever_card = RETRIEVER_CARD_TEMPLATE.render(
                retriever_type=self.__class__.__name__,
                version=__VERSION__,
                repo_path=self.cfg.retriever_path,
            )
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(retriever_card)

        # save the configuration
        cfg_path = os.path.join(retriever_path, "config.yaml")
        if not os.path.exists(cfg_path):
            with open(cfg_path, "w", encoding="utf-8") as f:
                OmegaConf.save(self.cfg, f)
        id_path = os.path.join(retriever_path, "cls.id")
        if not os.path.exists(id_path):
            with open(id_path, "w", encoding="utf-8") as f:
                f.write(self.__class__.__name__)
