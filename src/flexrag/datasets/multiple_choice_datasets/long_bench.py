import shutil
from typing import Annotated, Optional
from zipfile import ZipFile

from huggingface_hub import hf_hub_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, Context, configure

from ..reader import LineDelimitedReader
from .multiple_choice_dataset_base import (
    KNOWLEDGE_MULTIPLE_CHOICE_DATASETS,
    MULTIPLE_CHOICE_DATASETS,
    KnowledgeMultipleChoiceDatasetBase,
)


@configure
class LongBenchMCDatasetConfig:
    """Configuration for LongBenchMultipleChoiceDataset.

    `LongBench <https://arxiv.org/abs/2412.15204>`_ is a benchmark designed to evaluate
    the long-context understanding capabilities of large language models (LLMs).
    It features tasks that require processing and reasoning over extended contexts,
    pushing the boundaries of LLMs' abilities in handling long documents.

    Note that this dataset contains only two few-shot learning tasks: TREC and LSHT.

    :param data_path: The path to the LongBench dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param subset: The subset of LongBench to use. Default is `trec`.
        Available choices are: `trec`, and `lsht`.
    :type subset: str
    """

    data_path: Optional[str] = None
    subset: Annotated[str, Choices("trec", "lsht")] = "trec"


@MULTIPLE_CHOICE_DATASETS("long_bench", config_class=LongBenchMCDatasetConfig)
@KNOWLEDGE_MULTIPLE_CHOICE_DATASETS("long_bench", config_class=LongBenchMCDatasetConfig)
class LongBenchMCDataset(KnowledgeMultipleChoiceDatasetBase):
    _file_name_map = {"trec": "trec.jsonl", "lsht": "lsht.jsonl"}

    def __init__(self, config: LongBenchMCDatasetConfig):
        self._subset = config.subset
        # Download the dataset if not exists
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "long_bench"
        else:
            data_dir = config.data_path
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id="zai-org/LongBench",
                filename="data.zip",
                repo_type="dataset",
                local_dir=data_dir.as_posix(),
            )
            ZipFile((data_dir / "data.zip").as_posix()).extractall(data_dir.as_posix())
            # move the data to the data_dir
            source_dir = data_dir / "data"
            if source_dir.exists():
                for file in source_dir.iterdir():
                    shutil.move(file.as_posix(), data_dir.as_posix())
                source_dir.rmdir()
            (data_dir / "data.zip").unlink()

        # Load the dataset
        self._context_data = {}
        self._queries_data = {}
        self._answers_data = {}
        self._choices_data = {}
        self._qrels_data = {}
        data_path = data_dir / self._file_name_map[self._subset]
        reader = LineDelimitedReader(data_path)
        for item in reader:
            qid = item["_id"]
            self._queries_data[qid] = item["input"]
            self._answers_data[qid] = [
                item["all_classes"].index(ans) for ans in item["answers"]
            ]
            self._context_data[qid] = Context(
                context_id=qid,
                data={"text": item["context"]},
                source=f"LongBench-{self._subset}",
                meta_data={
                    "length": item.get("length", 0),
                    "language": item.get("language", "unknown"),
                },
            )
            self._choices_data[qid] = item["all_classes"]
            self._qrels_data[qid] = {qid: 1.0}
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
