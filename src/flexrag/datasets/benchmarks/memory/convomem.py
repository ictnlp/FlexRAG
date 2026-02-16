import json
import re
from pathlib import Path
from typing import Literal, Optional

from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure
from flexrag.common.dataclasses import ChatMessages

from ...core import DATASETS, MappingDataset, MultiSessionQASample


@configure
class ConvoMemDatasetConfig:
    """Configuration for ConvoMem.

    `ConvoMem <https://arxiv.org/abs/2511.10523>`_ This benchmark provides
    a large-scale, fine-grained evaluation of conversational memory with 75,336
    QA pairs spanning diverse memory types, enabling systematic analysis of
    memory growth, long-context limits, and trade-offs between full-context and
    retrieval-augmented approaches in long-term dialogue.

    :param data_path: The path to the ConvoMem dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param subset: The subset of the ConvoMem dataset to use. Default is "full".
        The available subsets are:
        - "abstention_evidence": Evaluates the model's ability to abstain from
            providing an answer when there is insufficient evidence.
        - "assistant_facts_evidence": Evaluates the model's ability to provide
            accurate answers based on the assistant's facts.
        - "changing_evidence": Evaluates the model's ability to adapt to changing
            evidence over time.
        - "implicit_connection_evidence": Evaluates the model's ability to make
            implicit connections between pieces of evidence.
        - "preference_evidence": Evaluates the model's ability to understand and
            incorporate user preferences into its answers.
        - "user_evidence": Evaluates the model's ability to provide accurate
            answers based on user-provided evidence.
        - "full": Evaluates the model's overall performance across all types of
            evidence.
    :type subset: Literal[
        "abstention_evidence",
        "assistant_facts_evidence",
        "changing_evidence",
        "implicit_connection_evidence",
        "preference_evidence",
        "user_evidence",
        "full",
    ]
    :param context_size: The number of conversation turns to include in the
        context for each QA pair. Default is 300. The available context sizes are:
        1, 2, 3, 4, 5, 6, 10, 20, 30, 50, 70, 100, 150, 200, and 300.
    :type context_size: int
    """

    data_path: Optional[str] = None
    subset: Literal[
        "abstention_evidence",
        "assistant_facts_evidence",
        "changing_evidence",
        "implicit_connection_evidence",
        "preference_evidence",
        "user_evidence",
        "full",
    ] = "full"
    context_size: int = 300


@DATASETS("convomem", config_class=ConvoMemDatasetConfig)
class ConvoMemDataset(MappingDataset[MultiSessionQASample]):
    def __init__(self, config: ConvoMemDatasetConfig):
        # set basic args
        self._subset = config.subset
        self._context_size = config.context_size
        assert self._context_size in {
            1,
            2,
            3,
            4,
            5,
            6,
            10,
            20,
            30,
            50,
            70,
            100,
            150,
            200,
            300,
        }, "Invalid context size."

        # prepare the data directory
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "convo_mem"
        else:
            data_dir = Path(config.data_path)
        if not data_dir.exists():
            data_dir.parent.mkdir(parents=True, exist_ok=True)

        # download the dataset
        if not data_dir.exists():
            snapshot_download(
                repo_id="Salesforce/ConvoMem",
                repo_type="dataset",
                local_dir=data_dir.as_posix(),
            )

        # load data
        self._data = []
        data_dir = data_dir / "core_benchmark" / "pre_mixed_testcases"
        if self._subset == "full":
            subsets = [
                "abstention_evidence",
                "assistant_facts_evidence",
                "changing_evidence",
                "implicit_connection_evidence",
                "preference_evidence",
                "user_evidence",
            ]
        else:
            subsets = [self._subset]
        for subset in subsets:
            subset_dir = data_dir / subset
            for subsub_dir in subset_dir.iterdir():
                evidence_num = re.match(r"(\d+)_evidence", subsub_dir.name)
                if evidence_num is None:
                    continue
                evidence_num = int(evidence_num.group(1))
                for file in subsub_dir.iterdir():
                    data = json.load(file.open())
                    for group in data:
                        # filter by context size
                        if group["contextSize"] != self._context_size:
                            continue
                        # construct context sessions
                        sessions = []
                        per_session_meta = []
                        for session in group["conversations"]:
                            session_data = []
                            for turn in session["messages"]:
                                session_data.append(
                                    {
                                        "role": turn["speaker"],
                                        "content": turn["text"],
                                    }
                                )
                            sessions.append(
                                ChatMessages.from_list(session_data, strict_mode=False)
                            )
                            per_session_meta.append(
                                {
                                    "session_id": session["id"],
                                    "contains_evidence": session["containsEvidence"],
                                    "generator": session["model_name"],
                                }
                            )
                        # construct QA samples
                        for item in group["evidenceItems"]:
                            question = item["question"]
                            answer = item["answer"]
                            golden_conv_ids = []
                            for session in item["conversations"]:
                                golden_conv_ids.append(session["id"])
                            metadata = {
                                "evidence_num": evidence_num,
                                "session_infos": per_session_meta,
                                "message_evidences": item["message_evidences"],
                                "category": item["category"],
                                "subset": subset,
                                "scenario_description": item["scenario_description"],
                                "person_id": item["personId"],
                                "core_model_name": item["core_model_name"],
                                "use_case_model_name": item["use_case_model_name"],
                                "golden_conv_ids": golden_conv_ids,
                            }
                            self._data.append(
                                MultiSessionQASample(
                                    question=question,
                                    answers=[answer],
                                    sessions=sessions,
                                    meta_data=metadata,
                                )
                            )
        return

    def __len__(self) -> int:
        return len(self._data)

    def get_item(self, index: int) -> MultiSessionQASample:
        return self._data[index]
