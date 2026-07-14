import json
from typing import Optional

from huggingface_hub import hf_hub_download

from flexrag.common import FLEXRAG_CACHE_DIR, Context, configure

from ...core import DATASETS, ContextualMCSample, MappingDataset


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


@DATASETS("long_bench_v2", config_class=LongBenchV2DatasetConfig)
class LongBenchV2Dataset(MappingDataset[ContextualMCSample]):
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
        self._context_data: dict[str, Context] = {}
        self._queries_data: dict[str, str] = {}
        self._answers_data: dict[str, list[int]] = {}
        self._choices_data: dict[str, list[str]] = {}
        self._metadata: dict[str, dict[str, str]] = {}
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
                metadata={"length": item["length"]},
            )
            self._choices_data[qid] = [item[key] for key in all_keys]
            self._metadata[qid] = {
                "domain": item.get("domain", "unknown"),
                "sub_domain": item.get("sub_domain", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
            }
        self._qids = list(self._queries_data.keys())
        return

    def __len__(self) -> int:
        return len(self._queries_data)

    def get_item(self, index: int) -> ContextualMCSample:
        qid = self._qids[index]
        return ContextualMCSample(
            question_id=qid,
            question=self._queries_data[qid],
            choices=self._choices_data[qid],
            answers=self._answers_data[qid],
            contexts=[self._context_data[qid]],
            metadata=self._metadata[qid],
        )
