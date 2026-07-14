from pathlib import Path
from typing import Annotated, Optional

from flexrag.common import FLEXRAG_CACHE_DIR, configure, download_and_extract
from flexrag.common.dataclasses import Context

from ...core import DATASETS, ContextualMCSample, MappingDataset
from ...reader import LineDelimitedReader


@configure
class QuALITYDatasetConfig:
    """Configuration for QuALITY Dataset.

    `QuALITY <https://arxiv.org/abs/2112.08608>`_ is a challenging multiple-choice
    dataset designed to evaluate reading comprehension and reasoning abilities over
    long-form text passages.

    :param data_path: The path to the QuALITY dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically to FLEXRAG_CACHE_DIR.
    :type data_path: str
    :param split: The dataset split to use. Default is `test`.
        Available choices are: `train`, `validation`, `test`.
    :type split: str
    :param html: Whether to use the HTML version of the dataset. Default is False.
    :type html: bool
    :param hard_only: Whether to load only the hard subset of the dataset. Default is False.
    :type hard_only: bool
    """

    data_path: Optional[str] = None
    split: Annotated[str, "train", "validation", "test"] = "test"
    html: bool = False
    hard_only: bool = False


RESOURCE_URL = "https://github.com/nyu-mll/quality/raw/refs/heads/main/data/v1.0.1/QuALITY.v1.0.1.zip"


@DATASETS("quality", config_class=QuALITYDatasetConfig)
class QuALITYDataset(MappingDataset[ContextualMCSample]):
    suffix_map = {
        "train": "train",
        "validation": "dev",
        "test": "test",
    }

    def __init__(self, config: QuALITYDatasetConfig):
        # download the dataset if not exists
        if config.data_path is not None:
            data_dir = Path(config.data_path)
        else:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "quality"
        if not data_dir.exists():
            data_dir.parent.mkdir(parents=True, exist_ok=True)
            download_and_extract(RESOURCE_URL, data_dir)

        # load the dataset
        self._context_data: dict[str, Context] = {}
        self._queries_data: dict[str, str] = {}
        self._answers_data: dict[str, list[int]] = {}
        self._choices_data: dict[str, list[str]] = {}
        self._qrels_data: dict[str, dict[str, float]] = {}
        suffix = self.suffix_map[config.split]
        if config.html:
            data_name = f"QuALITY.v1.0.1.{suffix}"
        else:
            data_name = f"QuALITY.v1.0.1.htmlstripped.{suffix}"
        reader = LineDelimitedReader(data_dir / data_name, file_format="jsonl")
        for row in reader:
            questions = row.pop("questions")
            context = Context(
                context_id=row.pop("article_id"),
                data={
                    "title": row.pop("title"),
                    "text": row.pop("article"),
                },
                source=row.pop("source"),
                metadata=row,
            )
            self._context_data[context.context_id] = context
            for q in questions:
                if config.hard_only and q["difficult"] == 0:
                    continue
                self._queries_data[q["question_unique_id"]] = q["question"]
                self._choices_data[q["question_unique_id"]] = q["options"]
                if "gold_label" in q:
                    self._answers_data[q["question_unique_id"]] = [q["gold_label"]]
                else:
                    self._answers_data[q["question_unique_id"]] = []
                self._qrels_data[q["question_unique_id"]] = {context.context_id: 1.0}
        self._qids = list(self._queries_data.keys())
        return

    def __len__(self) -> int:
        return len(self._queries_data)

    def get_item(self, index: int) -> ContextualMCSample:
        qid = self._qids[index]
        ctx_ids = list(self._qrels_data[qid].keys())
        return ContextualMCSample(
            question_id=qid,
            question=self._queries_data[qid],
            choices=self._choices_data[qid],
            answers=self._answers_data[qid],
            contexts=[self._context_data[ctx_id] for ctx_id in ctx_ids],
        )
