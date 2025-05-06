import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Generator, Iterable

import numpy as np
from omegaconf import OmegaConf

from flexrag.utils import LOGGER_MANAGER, Choices, ConfigureBase, SimpleProgressLogger

from .index_base import RetrieverIndexBase

logger = LOGGER_MANAGER.get_logger("flexrag.retriever.index.multi_field_index")


@dataclass
class MultiFieldIndexConfig(ConfigureBase):
    """Configuration for MultiFieldIndex.

    :param indexed_fields: Fields to be indexed.
        If more than one field is specified, each field will be processed separately and pointed to the same id.
    :type indexed_fields: list[str]
    :param merge_method: The method to merge the scores of the same context id.
        Available options are "max", "sum", "mean", and "concat".
        "max" will take the maximum score of the same context id.
        "sum" will take the sum of the scores of the same context id.
        "mean" will take the average of the scores of the same context id.
        "concat" will concatenate the texts of each field and index them together.
        Note that "concat" is only available if all indexed fields are of type str.
        If only one field is specified, this argument will be ignored.
        Defaults to "max".
    :type merge_method: str
    """

    indexed_fields: list[str]
    merge_method: Choices(["max", "sum", "mean", "concat"]) = "max"  # type: ignore


class MultiFieldIndex:
    """A wrapper index for multiple field contexts."""

    def __init__(self, cfg: MultiFieldIndexConfig, index: RetrieverIndexBase):
        # Initialize the MultiFieldIndex with a base index.
        self.index = index
        self.cfg = MultiFieldIndexConfig.extract(cfg)

        # load the context_id mapping if exists
        if self.index.cfg.index_path is not None:
            mapping_path = os.path.join(
                self.index.cfg.index_path, "context_mapping.pkl"
            )
            if os.path.exists(mapping_path):
                mapping = pickle.load(open(mapping_path, "rb"))
                self.context_id_to_index = mapping["context_id_to_index"]
                self.index_to_context_id = mapping["index_to_context_id"]
                self.max_field_num = mapping["max_field_num"]
            else:
                assert (
                    len(self.index) == 0
                ), "The index should be empty before building MultiFieldIndex."
        else:
            # check if the index is empty
            assert len(self.index) == 0, "The index should be empty before building."
            self.index_to_context_id: dict[int, str] = {}
            self.context_id_to_index: dict[str, list[int]] = defaultdict(list)
            self.max_field_num = 1

        # check consistency of the index
        assert len(self.index_to_context_id) == len(
            self.index
        ), "The length of the index and the context_id mapping should be the same."
        return

    def build_index(self, context_ids: Iterable[str], data: Iterable[dict[str, Any]]):
        """Build the index.
        The index will be serialized automatically if the `index_path` is set.

        :param context_ids: The context ids of the data.
        :type context_ids: Iterable[str]
        :param data: The data to build the index.
        :type data: Iterable[dict[str, Any]]
        :return: None
        """

        ctx_ids = []

        def get_data() -> Generator[Any | str, None, None]:
            """A helper function that yields the id or the data to be indexed."""
            for context_id, item in zip(context_ids, data):
                if self.cfg.indexed_fields is None:
                    indexed_fields = item.keys()
                else:
                    indexed_fields = [i for i in self.cfg.indexed_fields if i in item]
                if self.cfg.merge_method == "concat":
                    concat_str = ""
                    for field in indexed_fields:
                        assert isinstance(item[field], str)
                        concat_str += f"{field}: {item[field]} "
                    yield concat_str
                else:
                    for field in indexed_fields:
                        self.max_field_num = max(
                            self.max_field_num, len(indexed_fields)
                        )
                        ctx_ids.append(context_id)
                        yield item[field]

        self.index.build_index(get_data())

        # update the context_id mapping
        for n, context_id in enumerate(ctx_ids):
            self.context_id_to_index[context_id].append(n)
            self.index_to_context_id[n] = context_id

        # serialize the index if the `index_path` is set
        if self.index.cfg.index_path is not None:
            self.save_to_local()
        return

    def search_batch(
        self,
        query: list[Any],
        top_k: int,
        **search_kwargs,
    ) -> tuple[list[list[str]], np.ndarray]:
        """Search for the top_k most similar data indices to the query.
        This method will search the index in batches.

        :param query: The query data.
        :type query: list[Any]
        :param top_k: The number of most similar data indices to return, defaults to 10.
        :type top_k: int, optional
        :param batch_size: The batch size to search. Defaults to self.batch_size.
        :type batch_size: Optional[int]
        :param search_kwargs: Additional search arguments.
        :type search_kwargs: Any
        :return: The indices and scores of the top_k most similar data indices.
        :rtype: tuple[list[list[str]], np.ndarray]
        """

        def get_batch():
            """Yield data in batches."""
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
        batch_size = self.index.cfg.batch_size or self.index.cfg.batch_size
        total = len(query) if hasattr(query, "__len__") else None
        p_logger = SimpleProgressLogger(
            logger, total, interval=self.index.cfg.log_interval
        )
        for q in get_batch():
            r = self.search(q, top_k, **search_kwargs)
            indices.extend(r[0])
            scores.append(r[1])
            p_logger.update(step=batch_size, desc="Searching")
        return indices, np.concatenate(scores, axis=0)

    def search(
        self,
        query: list[Any],
        top_k: int,
        **search_kwargs,
    ) -> tuple[list[list[str]], np.ndarray]:
        """Search for the top_k most similar data indices to the query.

        :param query: The query data.
        :type query: list[Any]
        :param top_k: The number of most similar data indices to return, defaults to 10.
        :type top_k: int, optional
        :param search_kwargs: Additional search arguments.
        :type search_kwargs: Any
        :return: The indices and scores of the top_k most similar data indices.
        :rtype: tuple[list[list[str]], np.ndarray]
        """
        indices_batch, scores_batch = self.index.search(
            query, top_k * self.max_field_num, **search_kwargs
        )

        # convert the indices to context ids
        new_indices = []
        new_scores = []
        for indices, scores in zip(indices_batch, scores_batch):
            retrieved = defaultdict(list)
            # collect the scores for each context id
            for idx, score in zip(indices, scores):
                context_id = self.index_to_context_id[idx]
                retrieved[context_id].append(score)

            # merge the scores for each context id
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

            # sort the scores
            sorted_indices = sorted(retrieved.items(), key=lambda x: x[1], reverse=True)
            new_indices.append([x[0] for x in sorted_indices[:top_k]])
            new_scores.append([x[1] for x in sorted_indices[:top_k]])

        return new_indices, np.array(new_scores)

    def insert_batch(
        self,
        context_ids: Iterable[str],
        data: Iterable[dict[str, Any]],
        batch_size: int = None,
        serialize: bool = True,
    ) -> None:
        """Add data to the index in batches.
        This method will automatically perform the `serialize` method if the `index_path` is set.

        :param context_ids: The context ids of the data.
        :type context_ids: Iterable[str]
        :param data: The data to add.
        :type data: Iterable[dict[str, Any]]
        :param batch_size: The batch size to add data to the index. Defaults to self.batch_size.
        :type batch_size: int
        :param serialize: Whether to serialize the index after adding data. Defaults to True.
        :type serialize: bool
        :return: None
        """
        assert self.index.is_addable, "Current index is not addable."
        batch_size = batch_size or self.index.cfg.batch_size
        ctx_ids = []
        offset = len(self.index)

        def get_data_batch() -> Generator[list[Any], None, None]:
            """A helper function that yields data in batches."""
            batch = []
            for ctx_id, item in zip(context_ids, data):
                if self.cfg.indexed_fields is None:
                    indexed_fields = item.keys()
                else:
                    indexed_fields = self.cfg.indexed_fields
                for field in indexed_fields:
                    if field not in item:
                        continue
                    batch.append(item[field])
                    ctx_ids.append(ctx_id)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
            if batch:
                yield batch

        # iterate over the data in batches
        p_logger = SimpleProgressLogger(logger, interval=self.index.cfg.log_interval)
        for batch in get_data_batch():
            self.index.insert(batch)
            p_logger.update(step=len(batch), desc="Adding data")

        # update the context_id mapping
        for n, context_id in enumerate(ctx_ids):
            self.context_id_to_index[context_id].append(offset + n)
            self.index_to_context_id[offset + n] = context_id

        # serialize if the `index_path` is set
        if (self.index.cfg.index_path is not None) and serialize:
            self.save_to_local()
        return

    def insert(
        self,
        context_ids: list[str],
        data: list[dict[str, Any]],
        serialize: bool = True,
    ) -> None:
        """Add a batch of data to the index.

        :param context_ids: The context ids of the data.
        :type context_ids: list[str]
        :param data: The data to add.
        :type data: list[dict[str, Any]]
        :param serialize: Whether to serialize the index after adding data. Defaults to True.
        :type serialize: bool
        :return: None
        """
        assert len(context_ids) == len(
            data
        ), "The length of context_ids and data should be the same."
        assert self.index.is_addable, "Current index is not addable."

        # prepare the data
        batch = []
        ctx_ids = []
        for ctx_id, item in zip(context_ids, data):
            if self.cfg.indexed_fields is None:
                indexed_fields = item.keys()
            for field in indexed_fields:
                if field not in item:
                    continue
                batch.append(item)
                ctx_ids.append(ctx_id)

        # insert the data
        self.index.insert(batch)

        # update the context_id mapping
        for n, context_id in enumerate(ctx_ids):
            self.context_id_to_index[context_id].append(len(self.index) + n)
            self.index_to_context_id[len(self.index) + n] = context_id

        # serialize if the `index_path` is set
        if (self.index.cfg.index_path is not None) and serialize:
            self.save_to_local()
        return

    def clear(self) -> None:
        """Clear the index."""
        self.index_to_context_id.clear()
        self.context_id_to_index.clear()
        self.index.clear()
        return

    def save_to_local(self, index_path: str = None) -> None:
        """Serialize the index to the given path.

        :param index_path: The path to save the index. If None, the index will be saved to self.index.cfg.index_path.
        :type index_path: str
        :return: None
        """
        # check if the index_path is set
        index_path = index_path or self.index.cfg.index_path
        if index_path is None:
            raise ValueError("index_path is not set.")

        # serialize the index
        self.index.save_to_local(index_path)

        # serialize the configuration
        config_path = os.path.join(index_path, "multi_field_index_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            OmegaConf.save(self.cfg, f)

        # serialize the context_id mapping
        context_mapping_path = os.path.join(
            self.index.cfg.index_path, "context_mapping.pkl"
        )
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
    def load_from_local(index_path: str) -> "MultiFieldIndex":
        # load the index
        index = RetrieverIndexBase.load_from_local(index_path)

        # load the configuration
        config_path = os.path.join(index_path, "multi_field_index_config.yaml")
        assert os.path.exists(
            config_path
        ), f"Configuration file not found in {index_path}."
        cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(MultiFieldIndexConfig, cfg)

        return MultiFieldIndex(cfg, index)

    @property
    def is_addable(self) -> bool:
        """Check if the index is addable."""
        return self.index.is_addable

    def __len__(self) -> int:
        """Get the number of indexed contexts."""
        return len(self.context_id_to_index)
