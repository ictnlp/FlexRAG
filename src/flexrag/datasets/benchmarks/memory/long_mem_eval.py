import json
from pathlib import Path
from typing import Literal, Optional

from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, configure
from flexrag.common.dataclasses import ChatMessages

from ...core import DATASETS, MappingDataset, MultiSessionQASample


@configure
class LongMemEvalDatasetConfig:
    """Configuration for LongMemEval.

    `LongMemEval <http://arxiv.org/abs/2410.10813>`_ is a comprehensive benchmark
    for evaluating the long-term memory capabilities of LLM-based chat assistants
    across sustained multi-session interactions, covering information extraction,
    reasoning over time, knowledge updates, and reliable abstention.

    :param data_path: The path to the LongMemEval dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param split: The dataset split to use. Default is `s_cleaned`.
        Available choices are: `oracle`, `s_cleaned`, `m_cleaned`.
    :type split: Literal["oracle", "s_cleaned", "m_cleaned"]
    """

    data_path: Optional[str] = None
    split: Literal["oracle", "s_cleaned", "m_cleaned"] = "s_cleaned"


@DATASETS("long_mem_eval", config_class=LongMemEvalDatasetConfig)
class LongMemEvalDataset(MappingDataset[MultiSessionQASample]):
    def __init__(self, config: LongMemEvalDatasetConfig):
        # prepare the data directory
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "long_mem_eval"
        else:
            data_dir = Path(config.data_path)
        if not data_dir.exists():
            data_dir.parent.mkdir(parents=True, exist_ok=True)

        # download the dataset
        if not data_dir.exists():
            snapshot_download(
                repo_id="xiaowu0162/longmemeval-cleaned",
                local_dir=data_dir.as_posix(),
                repo_type="dataset",
            )

        # load the data items
        data_path = data_dir / f"longmemeval_{config.split}.json"
        raw_data = json.load(data_path.open(encoding="utf-8"))
        self._query_data = {}
        for item in raw_data:
            qid = item["question_id"]
            # parse sessions
            sessions = []
            for session, session_id, session_date in zip(
                item["haystack_sessions"],
                item["haystack_session_ids"],
                item["haystack_dates"],
                strict=True,
            ):
                sessions.append(
                    ChatMessages.from_list(
                        session,
                        strict_mode=False,
                        metadata={
                            "session_id": session_id,
                            "date": session_date,
                        },
                    )
                )
            # parse metadata
            metadata = {
                "abstention": qid.endswith("_abs"),
                "question_type": item["question_type"],
                "question_date": item["question_date"],
                "answer_session_ids": item["answer_session_ids"],
            }
            self._query_data[qid] = MultiSessionQASample(
                question_id=qid,
                sessions_id=qid,
                question=item["question"],
                answers=[str(item["answer"])],
                sessions=sessions,
                metadata=metadata,
            )
        return

    def __len__(self) -> int:
        return len(self._query_data)

    def get_item(self, index: int) -> MultiSessionQASample:
        qid = list(self._query_data.keys())[index]
        return self._query_data[qid]
