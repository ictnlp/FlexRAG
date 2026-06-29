import os
import shutil
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
from huggingface_hub import HfApi

from flexrag.common import (
    __VERSION__,
    FLEXRAG_CACHE_DIR,
    ProgressDisplay,
    SimpleProgressLogger,
    trace,
)
from flexrag.common.configure import configure, extract_config
from flexrag.common.dataclasses import Context, RetrievedContext

from ._index_state import METADATA_FILE, IndexState
from .database import LMDBRetrieverDatabase
from .index import ContextIndexBase
from .retriever_base import (
    DEFAULT_TOP_K,
    RETRIEVERS,
    RetrieverBase,
    RetrieverBaseConfig,
)


def _format_markdown_list(items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        return "- None"
    return "\n".join(f"- `{item}`" for item in items)


def _build_retriever_card(
    *,
    repo_id: str,
    version: str,
    context_count: int,
    fields: list[str],
    indexes: list[str],
    dirty_indexes: list[str],
) -> str:
    dirty_note = ""
    if dirty_indexes:
        dirty_note = (
            "\n## Dirty Indexes\n\n"
            "This artifact was uploaded with dirty indexes. Rebuild them before "
            "using those indexes for search if you need complete retrieval "
            "coverage.\n\n"
            f"{_format_markdown_list(dirty_indexes)}\n"
        )

    return f"""---
library_name: FlexRAG
tags:
- FlexRAG
- retrieval
- search
- RAG
---

# FlexRAG Retriever

This repository contains a FlexRAG `FlexRetriever` collection.

## Collection Summary

- FlexRAG version: `{version}`
- Context count: `{context_count}`

## Fields

{_format_markdown_list(fields)}

## Indexes

{_format_markdown_list(indexes)}
{dirty_note}
## Loading

```python
from flexrag.retrievers import FlexRetriever, FlexRetrieverConfig

retriever = FlexRetriever.from_hub(
    "{repo_id}",
    cfg=FlexRetrieverConfig(),
)
results = retriever.search("Who is Bruce Wayne?")
```
"""


@configure
class FlexRetrieverConfig(RetrieverBaseConfig):
    """Configuration for :class:`FlexRetriever`.

    The collection path is passed directly to :class:`FlexRetriever` and is
    intentionally not stored in config or metadata.

    :param batch_size: Batch size used by FlexRetriever itself when writing or
        copying context payloads, and by the inherited search method when
        splitting query batches. It does not override index build or insertion
        batch sizes. Defaults to ``32``.
    """


@RETRIEVERS("flex", config_class=FlexRetrieverConfig)
class FlexRetriever(RetrieverBase):
    """Local collection retriever with pluggable indexes.

    ``FlexRetriever`` stores contexts in a local collection and lets users add
    one or more named indexes over those contexts. It supports multi-field
    indexing through the index configuration, hybrid search across multiple
    indexes, incremental updates for addable indexes, dirty-index detection for
    non-addable indexes, and local or Hugging Face Hub artifact exchange.
    """

    def __init__(
        self,
        cfg: FlexRetrieverConfig,
        path: str | os.PathLike[str],
    ) -> None:
        cfg = extract_config(cfg, FlexRetrieverConfig)
        if cfg.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        super().__init__(cfg)
        self.root_path = os.fspath(path)
        self._prepare_collection()
        self.state = IndexState.load(self.root_path)
        self.database = LMDBRetrieverDatabase(self._database_path())
        self.index_table = self._load_indexes()
        self._check_consistency(allow_dirty=True)
        return

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | os.PathLike[str] = FLEXRAG_CACHE_DIR,
        cfg: FlexRetrieverConfig | None = None,
        **kwargs,
    ) -> "FlexRetriever":
        """Download a collection artifact from Hugging Face Hub and open it.

        :param repo_id: Hub repository ID.
        :param revision: Optional revision to download.
        :param token: Optional Hugging Face token.
        :param cache_dir: Local snapshot cache directory.
        :param cfg: Optional FlexRetriever runtime configuration.
        :param kwargs: Extra keyword arguments forwarded to
            :meth:`huggingface_hub.HfApi.snapshot_download`.
        :return: Opened retriever collection.
        """
        api = HfApi(token=token)
        repo_info = api.repo_info(repo_id)
        if repo_info is None:
            raise ValueError(f"Retriever {repo_id} not found on Hugging Face Hub.")
        repo_id = repo_info.id
        local_dir = os.path.join(cache_dir, repo_id.replace("/", "--"))
        snapshot = api.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            local_dir=local_dir,
            **kwargs,
        )
        if snapshot is None:
            raise RuntimeError(f"Retriever {repo_id} download failed.")
        return cls(cfg or FlexRetrieverConfig(), snapshot)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        token: str | None = os.environ.get("HF_TOKEN"),
        commit_message: str = "Update FlexRAG retriever",
        private: bool = False,
        allow_dirty: bool = False,
        **kwargs,
    ) -> str:
        """Upload this collection artifact to Hugging Face Hub.

        Dirty indexes are rejected by default because search semantics would be
        incomplete after download.

        :param repo_id: Hub repository ID to create or update.
        :param token: Optional Hugging Face token.
        :param commit_message: Hub commit message.
        :param private: Whether to create a private repository.
        :param allow_dirty: Whether to upload with dirty indexes.
        :param kwargs: Extra keyword arguments forwarded to
            :meth:`huggingface_hub.HfApi.upload_folder`.
        :return: Hub repository URL.
        """
        self._assert_pushable(allow_dirty=allow_dirty)
        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=repo_id,
            token=api.token,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=repo_url.repo_id,
            commit_message=commit_message,
            folder_path=self.root_path,
            **kwargs,
        )
        api.upload_file(
            repo_id=repo_url.repo_id,
            commit_message=commit_message,
            path_in_repo="README.md",
            path_or_fileobj=_build_retriever_card(
                repo_id=repo_url.repo_id,
                version=__VERSION__,
                context_count=self.count(),
                fields=self.fields,
                indexes=sorted(self.index_table),
                dirty_indexes=sorted(self.state.dirty_indexes),
            ).encode("utf-8"),
            repo_type="model",
        )
        return str(repo_url)

    @trace("retriever.flex_retriever.add_passages")
    def add_passages(
        self,
        passages: Iterable[Context],
        *,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        """Add contexts to the collection and update addable indexes.

        Non-addable indexes are marked dirty after new passages are inserted.
        Call :meth:`rebuild_index` before searching those indexes again.

        :param passages: Contexts to add. Every context must provide a unique
            ``context_id``.
        :param log_interval: Number of inserted contexts between progress
            updates.
        :param display: Progress display mode.
        :return: None.
        """
        new_ids: list[str] = []
        new_data: list[dict[str, Any]] = []
        for context in passages:
            if context.context_id is None:
                raise ValueError("context_id is required.")
            if context.context_id in self.database or context.context_id in new_ids:
                raise ValueError(f"Duplicate context_id: {context.context_id}")
            new_ids.append(context.context_id)
            new_data.append(dict(context.data))
        if not new_ids:
            return

        with SimpleProgressLogger(
            None,
            interval=log_interval,
            display=display,
        ) as p_logger:
            for start in range(0, len(new_ids), self.cfg.batch_size):
                ids = new_ids[start : start + self.cfg.batch_size]
                data = new_data[start : start + self.cfg.batch_size]
                self.database[ids] = data
                p_logger.update(step=len(ids), desc="Adding passages")

        for index_name, index in self.index_table.items():
            if index.is_addable:
                try:
                    index.insert_batch(
                        new_ids,
                        (self.database[context_id] for context_id in new_ids),
                        log_interval=log_interval,
                        display=display,
                    )
                    self._save_index(index_name, index)
                    self.state.mark_clean(index_name)
                except Exception:
                    self.state.mark_dirty(index_name)
                    self._save_state()
                    raise
            else:
                self.state.mark_dirty(index_name)
        self._save_state()
        return

    @trace("retriever.flex_retriever.add_index")
    def add_index(self, index_name: str, index: ContextIndexBase) -> None:
        """Add a clean index built over the full collection.

        :param index_name: Unique index name inside this collection.
        :param index: Context index instance to build and store.
        :return: None.
        """
        if index_name in self.index_table:
            raise ValueError(f"Index already exists: {index_name}")
        if len(self.database) > 0:
            self._build_index(index_name, index)
        self._save_index(index_name, index)
        self.index_table[index_name] = index
        self.state.add_index(index_name)
        self._save_state()
        return

    @trace("retriever.flex_retriever.rebuild_index")
    def rebuild_index(self, index_name: str | None = None) -> None:
        """Rebuild one index or all indexes from the full database.

        :param index_name: Optional index name. If omitted, all indexes are
            rebuilt.
        :return: None.
        """
        index_names = [index_name] if index_name is not None else list(self.index_table)
        for name in index_names:
            if name not in self.index_table:
                raise KeyError(f"Index not found: {name}")
            index = self.index_table[name]
            self._build_index(name, index)
            self._save_index(name, index)
            self.state.mark_clean(name)
        self._save_state()
        return

    def remove_index(self, index_name: str) -> None:
        """Remove one index from the collection.

        :param index_name: Index name.
        :return: None.
        """
        if index_name not in self.index_table:
            raise KeyError(f"Index not found: {index_name}")
        index = self.index_table.pop(index_name)
        index.clear()
        index_path = self._index_path(index_name)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        self.state.remove_index(index_name)
        self._save_state()
        return

    @trace("retriever.flex_retriever.search")
    def _search(
        self,
        query: list[str],
        *,
        top_k: int = DEFAULT_TOP_K,
        used_indexes: list[str] | None = None,
        indexes_merge_method: str = "rrf",
        indexes_merge_weights: list[float] | None = None,
        rrf_base: int = 60,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        if used_indexes is None:
            used_indexes = list(self.index_table)
        if not used_indexes:
            raise ValueError("No indexes are available for search.")
        dirty = sorted(set(used_indexes) & set(self.state.dirty_indexes))
        if dirty:
            raise RuntimeError(
                "Cannot search dirty indexes. Rebuild first: " + ", ".join(dirty)
            )
        for index_name in used_indexes:
            if index_name not in self.index_table:
                raise KeyError(f"Index not found: {index_name}")

        all_context_ids: list[list[list[str]]] = []
        all_scores: list[np.ndarray] = []
        for index_name in used_indexes:
            context_ids, scores = self.index_table[index_name].search(
                query,
                top_k,
                **search_kwargs,
            )
            all_context_ids.append(context_ids)
            all_scores.append(scores)

        merged_ids, merged_scores = self._merge_results(
            used_indexes,
            all_context_ids,
            all_scores,
            top_k,
            indexes_merge_method,
            indexes_merge_weights,
            rrf_base,
        )
        return self._build_results(query, merged_ids, merged_scores)

    def get(self, context_id: str) -> Context:
        """Get one context by ID.

        :param context_id: Context ID.
        :return: Retrieved context payload.
        """
        return Context(
            context_id=context_id,
            data=dict(self.database[context_id]),
            source=self.root_path,
        )

    def count(self) -> int:
        """Return the number of contexts in the collection.

        :return: Context count.
        """
        return len(self.database)

    def clear(self) -> None:
        """Clear contexts and all index contents.

        The collection artifact remains on disk, but the database and active
        indexes become empty and clean.

        :return: None.
        """
        self.database.clear()
        for index_name, index in self.index_table.items():
            index.clear()
            self._save_index(index_name, index)
            self.state.mark_clean(index_name)
        self._save_state()
        return

    def export_to(
        self,
        path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> None:
        """Export the current on-disk collection to another path.

        The current object remains attached to its original path.

        :param path: Target directory.
        :param overwrite: Whether to remove an existing target directory.
        :return: None.
        """
        root = os.fspath(path)
        if os.path.exists(root):
            if not overwrite and os.listdir(root):
                raise FileExistsError(f"Export path is not empty: {root}")
            if overwrite:
                shutil.rmtree(root)
        os.makedirs(root, exist_ok=True)

        state = IndexState.create()
        state.indexes = sorted(self.index_table)
        state.dirty_indexes = sorted(self.state.dirty_indexes)
        state.save(root)

        target_db = LMDBRetrieverDatabase(self._database_path(root))
        try:
            self._copy_database(target_db)
        finally:
            target_db.close()

        os.makedirs(os.path.join(root, "indexes"), exist_ok=True)
        for index_name, index in self.index_table.items():
            self._save_index_to_root(root, index_name, index)
        return

    @property
    def fields(self) -> list[str]:
        """Return stored context fields.

        :return: Context field names.
        """
        return self.database.fields

    def close(self) -> None:
        """Close the underlying LMDB database.

        :return: None.
        """
        self.database.close()
        return

    def _cache_fingerprint(self) -> dict[str, Any]:
        return {
            "path": os.path.abspath(self.root_path),
            "updated_at": self.state.updated_at,
        }

    def _build_index(self, index_name: str, index: ContextIndexBase) -> None:
        scratch_path = self._scratch_path(index_name)
        if os.path.exists(scratch_path):
            shutil.rmtree(scratch_path)
        os.makedirs(scratch_path, exist_ok=True)
        try:
            index.build_index(
                self.database.ids,
                self.database.values(),
                scratch_path=scratch_path,
            )
        finally:
            if os.path.exists(scratch_path):
                shutil.rmtree(scratch_path)
        return

    def _load_indexes(self) -> dict[str, ContextIndexBase]:
        indexes = {}
        for index_name in self.state.indexes:
            index_path = self._index_path(index_name)
            if os.path.exists(index_path):
                indexes[index_name] = ContextIndexBase.load_from_local(index_path)
            else:
                self.state.mark_dirty(index_name)
        self._cleanup_staging()
        self._save_state()
        return indexes

    def _merge_results(
        self,
        used_indexes: list[str],
        all_context_ids: list[list[list[str]]],
        all_scores: list[np.ndarray],
        top_k: int,
        merge_method: str,
        merge_weights: list[float] | None,
        rrf_base: int,
    ) -> tuple[list[list[str]], list[list[float]]]:
        if len(all_scores) == 1:
            return all_context_ids[0], all_scores[0].tolist()

        if merge_weights is not None:
            if len(merge_weights) != len(used_indexes):
                raise ValueError("indexes_merge_weights length mismatch.")
            total_weight = sum(merge_weights)
            merge_weights = [weight / total_weight for weight in merge_weights]
        else:
            merge_weights = [1.0 / len(all_scores)] * len(all_scores)

        merged_ids: list[list[str]] = []
        merged_scores: list[list[float]] = []
        match merge_method:
            case "rrf":
                for query_idx in range(len(all_context_ids[0])):
                    scores_dict: dict[str, float] = defaultdict(float)
                    for ctx_ids, scores, weight in zip(
                        all_context_ids,
                        all_scores,
                        merge_weights,
                    ):
                        sort_ranks = scores[query_idx].argsort()[::-1] + 1
                        for ctx_id, rank in zip(ctx_ids[query_idx], sort_ranks):
                            scores_dict[ctx_id] += weight / (rank + rrf_base)
                    sorted_items = sorted(scores_dict.items(), key=lambda item: -item[1])
                    merged_ids.append([item[0] for item in sorted_items[:top_k]])
                    merged_scores.append([item[1] for item in sorted_items[:top_k]])
            case "linear":
                normalized_scores = list(all_scores)
                use_infimum = all(
                    self.index_table[index_name].infimum != float("-inf")
                    for index_name in used_indexes
                )
                for idx, index_name in enumerate(used_indexes):
                    if use_infimum:
                        infimum = self.index_table[index_name].infimum
                    else:
                        infimum = normalized_scores[idx].min(axis=1, keepdims=True)
                    denominator = normalized_scores[idx].max(
                        axis=1,
                        keepdims=True,
                    ) - infimum
                    denominator[denominator == 0] = 1
                    normalized_scores[idx] = (normalized_scores[idx] - infimum) / (
                        denominator
                    )
                for query_idx in range(len(all_context_ids[0])):
                    scores_dict = defaultdict(float)
                    for ctx_ids, scores, weight in zip(
                        all_context_ids,
                        normalized_scores,
                        merge_weights,
                    ):
                        for ctx_id, score in zip(ctx_ids[query_idx], scores[query_idx]):
                            scores_dict[ctx_id] += float(score) * weight
                    sorted_items = sorted(scores_dict.items(), key=lambda item: -item[1])
                    merged_ids.append([item[0] for item in sorted_items[:top_k]])
                    merged_scores.append([item[1] for item in sorted_items[:top_k]])
            case _:
                raise ValueError(f"Unknown merge method: {merge_method}")
        return merged_ids, merged_scores

    def _build_results(
        self,
        queries: list[str],
        merged_ids: list[list[str]],
        merged_scores: list[list[float]],
    ) -> list[list[RetrievedContext]]:
        results: list[list[RetrievedContext]] = []
        for query, context_ids, scores in zip(queries, merged_ids, merged_scores):
            data = self.database[context_ids] if context_ids else []
            results.append(
                [
                    RetrievedContext(
                        context_id=context_id,
                        retriever="FlexRetriever",
                        query=query,
                        score=float(score),
                        data=dict(context_data),
                    )
                    for context_id, score, context_data in zip(
                        context_ids,
                        scores,
                        data,
                    )
                ]
            )
        return results

    def _check_consistency(self, *, allow_dirty: bool = False) -> None:
        for index_name, index in self.index_table.items():
            if allow_dirty and index_name in self.state.dirty_indexes:
                continue
            if len(index) != len(self.database):
                raise RuntimeError(
                    f"Index/database size mismatch for {index_name}: "
                    f"{len(index)} != {len(self.database)}"
                )
        return

    def _database_path(self, root: str | None = None) -> str:
        return os.path.join(root or self.root_path, "database.lmdb")

    def _index_path(self, index_name: str, root: str | None = None) -> str:
        return os.path.join(root or self.root_path, "indexes", index_name)

    def _scratch_path(self, index_name: str) -> str:
        return os.path.join(self.root_path, "indexes", ".scratch", index_name)

    def _save_index(self, index_name: str, index: ContextIndexBase) -> None:
        self._save_index_to_root(self.root_path, index_name, index)
        return

    def _save_index_to_root(
        self,
        root: str,
        index_name: str,
        index: ContextIndexBase,
    ) -> None:
        staging_path = os.path.join(root, "indexes", ".staging", index_name)
        final_path = os.path.join(root, "indexes", index_name)
        if os.path.exists(staging_path):
            shutil.rmtree(staging_path)
        os.makedirs(os.path.dirname(staging_path), exist_ok=True)
        index.save_to_local(staging_path)
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        os.replace(staging_path, final_path)
        return

    def _save_state(self) -> None:
        self.state.save(self.root_path)
        return

    def _cleanup_staging(self) -> None:
        staging_root = os.path.join(self.root_path, "indexes", ".staging")
        if os.path.exists(staging_root):
            shutil.rmtree(staging_root)
        return

    def _prepare_collection(self) -> None:
        if os.path.isfile(self.root_path):
            raise NotADirectoryError(f"Collection path is a file: {self.root_path}")
        metadata_path = os.path.join(self.root_path, METADATA_FILE)
        if os.path.exists(metadata_path):
            return
        if os.path.isdir(self.root_path) and os.listdir(self.root_path):
            raise FileExistsError(
                f"Collection path is non-empty but has no {METADATA_FILE}: "
                f"{self.root_path}"
            )
        os.makedirs(self.root_path, exist_ok=True)
        os.makedirs(os.path.join(self.root_path, "indexes"), exist_ok=True)
        IndexState.create().save(self.root_path)
        return

    def _assert_pushable(self, *, allow_dirty: bool) -> None:
        if self.state.dirty_indexes and not allow_dirty:
            dirty_indexes = ", ".join(self.state.dirty_indexes)
            raise RuntimeError(
                "Cannot push collection with dirty indexes. "
                f"Rebuild first or pass allow_dirty=True: {dirty_indexes}"
            )
        return

    def _copy_database(self, target_db: LMDBRetrieverDatabase) -> None:
        ids = list(self.database.ids)
        for start in range(0, len(ids), self.cfg.batch_size):
            batch_ids = ids[start : start + self.cfg.batch_size]
            target_db[batch_ids] = self.database[batch_ids]
        return
