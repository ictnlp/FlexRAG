from pathlib import Path
from typing import Annotated, Literal, Optional

from datasets import load_dataset
from dill.tests.test_registered import q
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, ChatTurn, Choices, configure
from flexrag.common.dataclasses import ChatMessages, Context

from ...core import (
    DATASETS,
    ContextualQASample,
    MappingDataset,
    MultiSessionQASample,
)


@configure
class MemoryAgentBenchDatasetConfig:
    """Configuration for MemoryAgentBench.

    `MemoryAgentBench <http://arxiv.org/abs/2410.10813>`_ is a comprehensive multi-turn
    benchmark for evaluating memory agents, systematically assessing four core memory
    competencies—accurate retrieval, test-time learning, long-range understanding, and
    selective forgetting—under realistic, incremental interaction settings.

    :param data_path: The path to the MemoryAgentBench dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param split: The dataset split to use. Default is `Accurate_Retrieval`.
        Available choices are:
        `Accurate_Retrieval`, `Test_Time_Learning`,
        `Long_Range_Understanding`, `Conflict_Resolution`.
    :type split: Literal[
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Conflict_Resolution",
    ]
    :param parse_contexts: Whether to parse contexts into structured format.
        Default is False.
    :type parse_contexts: bool
    """

    data_path: Optional[str] = None
    split: Literal[
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Conflict_Resolution",
    ] = "Accurate_Retrieval"
    parse_contexts: bool = False


@DATASETS("memory_agent_bench", config_class=MemoryAgentBenchDatasetConfig)
class MemoryAgentBenchDataset(
    MappingDataset[MultiSessionQASample | ContextualQASample]
):
    def __init__(self, config: MemoryAgentBenchDatasetConfig):
        self._parse_contexts = config.parse_contexts
        # prepare the data directory
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "memory_agent_bench"
        else:
            data_dir = Path(config.data_path)
        if not data_dir.exists():
            data_dir.parent.mkdir(parents=True, exist_ok=True)

        # download the dataset
        if not data_dir.exists():
            snapshot_download(
                repo_id="ai-hyz/MemoryAgentBench",
                local_dir=data_dir.as_posix(),
                repo_type="dataset",
            )

        # load the data items
        raw_data = load_dataset(
            data_dir.as_posix(),
            split=config.split,
        )
        self._query_data = {}
        self._contexts = {}
        for i, item in enumerate(raw_data):
            # parse contexts
            context_id = f"{config.split}_context_{i}"
            if self._parse_contexts:
                raise NotImplementedError
            else:
                self._contexts[context_id] = Context(
                    context_id=context_id,
                    data={"text": item["context"]},
                )
            # parse qa pairs
            for j, qid in enumerate(item["metadata"]["qa_pair_ids"]):
                # parse question and answer
                question, answer = item["questions"][j], item["answers"][j]
                # parse metadata
                metadata = {"source": item["metadata"]["source"]}
                if item["metadata"].get("question_ids", None) is not None:
                    metadata["question_id"] = item["metadata"]["question_ids"][j]
                if item["metadata"].get("question_types", None) is not None:
                    metadata["question_type"] = item["metadata"]["question_types"][j]
                if item["metadata"].get("question_dates", None) is not None:
                    metadata["question_date"] = item["metadata"]["question_dates"][j]
                if item["metadata"].get("demo", None) is not None:
                    metadata["demo"] = item["metadata"]["demo"]
                if item["metadata"].get("keypoints", None) is not None:
                    metadata["keypoints"] = item["metadata"]["keypoints"]
                if item["metadata"].get("previous_events", None) is not None:
                    metadata["previous_events"] = item["metadata"]["previous_events"]
                if item["metadata"].get("haystack_sessions", None) is not None:
                    metadata["haystack_sessions"] = item["metadata"][
                        "haystack_sessions"
                    ]
                self._query_data[qid] = {
                    "question": question,
                    "answer": answer,
                    "meta_data": metadata,
                    "context_id": context_id,
                }
        return

    def __len__(self) -> int:
        return len(self._query_data)

    def get_item(self, index: int) -> MultiSessionQASample | ContextualQASample:
        qid = list(self._query_data.keys())[index]
        data = self._query_data[qid]
        context = self._contexts[data["context_id"]]
        if isinstance(context, Context):
            return ContextualQASample(
                question=data["question"],
                answers=data["answer"],
                contexts=[context],
                question_id=qid,
                meta_data=data["meta_data"],
            )
        raise NotImplementedError
