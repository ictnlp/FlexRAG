import shutil
from typing import Annotated, Optional
from zipfile import ZipFile

from huggingface_hub import hf_hub_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, Context, configure

from ..reader import LineDelimitedReader
from .qa_dataset_base import KNOWLEDGE_QA_DATASETS, QA_DATASETS, KnowledgeQADatasetBase


@configure
class LongBenchQADatasetConfig:
    """Configuration for LongBenchQADataset.

    `LongBench <https://arxiv.org/abs/2308.14508>`_ is a benchmark designed to evaluate
    the long-context understanding capabilities of large language models (LLMs).
    It features tasks that require processing and reasoning over extended contexts,
    pushing the boundaries of LLMs' abilities in handling long documents.

    Note that TREC and LSHT subsets are contained in the Multiple Choice Dataset variant.

    :param data_path: The path to the LongBench dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param subset: The subset of LongBench to use. Default is `narrative_qa`.
        Available choices are:

        - Single Document QA Tasks: `narrative_qa`, `qasper`, `multifield_qa_en`, `multifield_qa_zh`,
        - Multi Document QA Tasks: `hotpot_qa`, `2wikimultihop_qa`, `musique`, `dureader`.
        - Summarization Tasks: `gov_report`, `qm_sum`, `multi_news`, `vc_sum`.
        - Few-shot Learning Tasks: `trec`, `trivia_qa`, `sam_sum`, `lsht`.
        - Synthetic Tasks: `passage_count`, `passage_retrieval_en`, `passage_retrieval_zh`.
        - Code Completion Tasks: `lcc`, `repobench_p`.
    :type subset: str
    """

    data_path: Optional[str] = None
    subset: Annotated[
        str,
        Choices(
            "narrative_qa",
            "qasper",
            "multifield_qa_en",
            "multifield_qa_zh",
            "hotpot_qa",
            "2wikimultihop_qa",
            "musique",
            "dureader",
            "gov_report",
            "qm_sum",
            "multi_news",
            "vc_sum",
            "trec",
            "trivia_qa",
            "sam_sum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench_p",
        ),
    ] = "narrative_qa"


@QA_DATASETS("long_bench", config_class=LongBenchQADatasetConfig)
@KNOWLEDGE_QA_DATASETS("long_bench", config_class=LongBenchQADatasetConfig)
class LongBenchQADataset(KnowledgeQADatasetBase):
    _file_name_map = {
        "narrative_qa": "narrativeqa.jsonl",
        "qasper": "qasper.jsonl",
        "multifield_qa_en": "multifieldqa_en.jsonl",
        "multifield_qa_zh": "multifieldqa_zh.jsonl",
        "hotpot_qa": "hotpotqa.jsonl",
        "2wikimultihop_qa": "2wikimqa.jsonl",
        "musique": "musique.jsonl",
        "dureader": "dureader.jsonl",
        "gov_report": "gov_report.jsonl",
        "qm_sum": "qmsum.jsonl",
        "multi_news": "multi_news.jsonl",
        "vc_sum": "vcsum.jsonl",
        "trivia_qa": "triviaqa.jsonl",
        "sam_sum": "samsum.jsonl",
        "passage_count": "passage_count.jsonl",
        "passage_retrieval_en": "passage_retrieval_en.jsonl",
        "passage_retrieval_zh": "passage_retrieval_zh.jsonl",
        "lcc": "lcc.jsonl",
        "repobench_p": "repobench-p.jsonl",
    }

    def __init__(self, config: LongBenchQADatasetConfig):
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
        self._qrels_data = {}
        self._meta_data = {}
        data_path = data_dir / self._file_name_map[self._subset]
        reader = LineDelimitedReader(data_path)
        for item in reader:
            qid = item["_id"]
            self._queries_data[qid] = item["input"]
            self._answers_data[qid] = item["answers"]
            self._context_data[qid] = Context(
                context_id=qid,
                data={"text": item["context"]},
                source=f"LongBench-{self._subset}",
                meta_data={
                    "length": item.get("length", 0),
                    "language": item.get("language", "unknown"),
                },
            )
            # all classes for classification tasks (trec, lsht)
            self._meta_data[qid] = {"all_classes": item.get("all_classes", [])}
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
