import json
from typing import Optional

from huggingface_hub import hf_hub_download

from flexrag.common import FLEXRAG_CACHE_DIR, Context, configure

from .multiple_choice_dataset_base import (
    KNOWLEDGE_MULTIPLE_CHOICE_DATASETS,
    MULTIPLE_CHOICE_DATASETS,
    KnowledgeMultipleChoiceDatasetBase,
)


@configure
class LongBenchV2DatasetConfig:
    """Configuration for LongBenchV2Dataset.

    `LongBenchV2 <https://arxiv.org/abs/2412.15204>`_ is a benchmark designed to evaluate
    the long-context understanding capabilities of large language models (LLMs).
    It features tasks that require processing and reasoning over extended contexts,
    pushing the boundaries of LLMs' abilities in handling long documents.

    :param data_path: The path to the LongBenchV2 dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    """

    data_path: Optional[str] = None


@MULTIPLE_CHOICE_DATASETS("long_bench_v2", config_class=LongBenchV2DatasetConfig)
@KNOWLEDGE_MULTIPLE_CHOICE_DATASETS("long_bench_v2", config_class=LongBenchV2DatasetConfig)  # fmt: skip
class LongBenchV2Dataset(KnowledgeMultipleChoiceDatasetBase):
    def __init__(self, config: LongBenchV2DatasetConfig):
        # Download the dataset if not exists
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "long_bench_v2"
        else:
            data_dir = config.data_path
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id="zai-org/LongBench-v2",
                filename="data.json",
                repo_type="dataset",
                local_dir=data_dir.as_posix(),
            )

        # Load the dataset
        self._context_data = {}
        self._queries_data = {}
        self._answers_data = {}
        self._choices_data = {}
        self._qrels_data = {}
        self._meta_data = {}
        data_path = data_dir / "data.json"
        data = json.load(open(data_path, "r", encoding="utf-8"))
        all_keys = ("choice_A", "choice_B", "choice_C", "choice_D")
        for item in data:
            qid = item["_id"]
            self._queries_data[qid] = item["question"]
            self._answers_data[qid] = [ord(item["answer"]) - ord("A")]
            self._context_data[qid] = Context(
                context_id=qid,
                data={"text": item["context"]},
                source=f"LongBench-v2",
                meta_data={"length": item["length"]},
            )
            self._choices_data[qid] = [item[key] for key in all_keys]
            self._qrels_data[qid] = {qid: 1.0}
            self._meta_data[qid] = {
                "domain": item.get("domain", "unknown"),
                "sub_domain": item.get("sub_domain", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
            }
        return

    @property
    def _queries(self) -> dict[str, str]:
        return self._queries_data

    @property
    def _answers(self) -> dict[str, list[str]] | None:
        return self._answers_data

    @property
    def _qrels(self) -> dict[str, dict[str, float]]:
        return self._qrels_data

    @property
    def _contexts(self) -> dict[str, Context]:
        return self._context_data

    @property
    def _choices(self) -> dict[str, list[str]]:
        return self._choices_data
