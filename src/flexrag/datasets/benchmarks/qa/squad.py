import os
from pathlib import Path
from typing import Annotated, Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, configure
from flexrag.common.dataclasses import Context

from ...core import DATASETS, ContextualQASample, MappingDataset


@configure
class SQuADDatasetConfig:
    """Configuration for SQuADDataset.

    `SQuAD <https://arxiv.org/abs/1606.05250>`_ and `SQuAD 2.0 <https://arxiv.org/abs/1806.03822>`_
    is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set
    of Wikipedia articles, where the answer to every question is a segment of text, or span,
    from the corresponding reading passage, or the question might be unanswerable.

    :param data_path: The path to the SQuAD dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param split: The dataset split to use. Default is `validation`.
        Available choices are: `train`, `validation`.
    :type split: str
    :param version: The version of the SQuAD dataset to use. Default is `v2.0`.
        Available choices are: `v1.1`, `v2.0`.
    :type version: str
    """

    data_path: Optional[str] = None
    split: Annotated[str, Choices("train", "validation")] = "validation"
    version: Annotated[str, Choices("v1.1", "v2.0")] = "v2.0"


@DATASETS("squad", config_class=SQuADDatasetConfig)
class SQuADDataset(MappingDataset[ContextualQASample]):
    def __init__(self, config: SQuADDatasetConfig):
        # Download the dataset if not exists
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "squad"
        else:
            data_dir = Path(config.data_path)
        data_path = data_dir / config.version
        if not data_path.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            if config.version == "v2.0":
                repo_id = "rajpurkar/squad_v2"
            else:
                repo_id = "rajpurkar/squad"
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=data_path.as_posix(),
                token=os.getenv("HF_TOKEN"),
            )
        data = load_dataset(data_path.as_posix(), split=config.split)

        self._context_data = {}
        self._queries_data = {}
        self._answers_data = {}
        self._qrels_data = {}
        for item in data:
            self._queries_data[item["id"]] = item["question"]
            self._answers_data[item["id"]] = item["answers"]["text"]
            context = Context(
                context_id=item["id"],
                data={
                    "text": item["context"],
                    "title": item["title"],
                },
                source="squad",
            )
            self._context_data[item["id"]] = context
            self._qrels_data[item["id"]] = {item["id"]: 1.0}
        self._qids = list(self._queries_data.keys())
        return

    def __len__(self) -> int:
        return len(self._queries_data)

    def get_item(self, index: int) -> ContextualQASample:
        qid = self._qids[index]
        contexts = [
            self._context_data[ctx_id] for ctx_id in self._qrels_data[qid].keys()
        ]
        return ContextualQASample(
            question=self._queries_data[qid],
            question_id=qid,
            contexts=contexts,
            answers=self._answers_data[qid],
        )
