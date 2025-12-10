import os
from pathlib import Path
from typing import Annotated, Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, configure
from flexrag.common.dataclasses import Context
from flexrag.common.logging import LOGGER_MANAGER
from flexrag.common.misc import download

from ..reader import LineDelimitedReader
from .qa_dataset_base import KNOWLEDGE_QA_DATASETS, QA_DATASETS, KnowledgeQADatasetBase

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.kilt_qa")


@configure
class KiltQADatasetConfig:
    """Configuration for KiltQADataset.

    `KiltQA <https://arxiv.org/abs/2009.02252>`_ is subsets of KILT benchmark
    designed for open-domain question answering. It includes several datasets such as
    HotpotQA, Natural Questions, TriviaQA, and ELI5.

    :param data_path: The path to the KiltQA dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param subset: The subset of KiltQA to use. Default is `nq`.
        Available choices are:

        - `hotpotqa`: HotpotQA subset.
        - `nq`: Natural Questions subset.
        - `triviaqa`: TriviaQA subset.
        - `eli5`: ELI5 subset.
    :type subset: str
    :param split: The data split to use. Default is `validation`.
        Available choices are:

        - `train`: Training split.
        - `validation`: Validation split.
        - `test`: Test split.
    :type split: str
    :param load_corpus: Whether to load the full corpus for retrieval. Default is True
    :type load_corpus: bool
    :param corpus_path: The path to the corpus file. Default is None.
        If not provided, the corpus will be downloaded automatically to the `data_path`.
    :type corpus_path: Optional[str]
    :param triviaqa_path: The path to the TriviaQA dataset file. Default is None.
        As the KILT release does not include the full TriviaQA dataset, the original
        TriviaQA dataset needs to be downloaded separately. If not provided, the
        dataset will be downloaded automatically to the cache directory.
    :type triviaqa_path: Optional[str]
    """

    data_path: Optional[str] = None
    subset: Annotated[str, Choices("hotpotqa", "nq", "triviaqa", "eli5")] = "nq"
    split: Annotated[str, Choices("train", "validation", "test")] = "validation"
    load_corpus: bool = True
    corpus_path: Optional[str] = None
    triviaqa_path: Optional[str] = None


CORPUS_URL = "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"


@QA_DATASETS("kilt_qa", config_class=KiltQADatasetConfig)
@KNOWLEDGE_QA_DATASETS("kilt_qa", config_class=KiltQADatasetConfig)
class KiltQADataset(KnowledgeQADatasetBase):
    def __init__(self, config: KiltQADatasetConfig):
        self._subset = config.subset
        self._split = config.split
        # download the kilt dataset if not exists
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "kilt"
        else:
            data_dir = Path(config.data_path)
        if not data_dir.exists():
            data_dir.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="facebook/kilt_tasks",
                repo_type="dataset",
                local_dir=data_dir.as_posix(),
                token=os.getenv("HF_TOKEN"),
            )
        # load the corpus if needed
        if config.load_corpus:
            if config.corpus_path is not None:
                corpus_path = Path(config.corpus_path)
            else:
                corpus_path = data_dir / "kilt_knowledgesource.json"
            if not corpus_path.exists():
                download(CORPUS_URL, corpus_path)
            self._context_data = self._load_corpus(corpus_path)
        else:
            self._context_data = {}

        # load the QA pairs
        if self._subset == "triviaqa":
            data_name = "triviaqa_support_only"
        else:
            data_name = self._subset
        subset = load_dataset(data_dir.as_posix(), name=data_name, split=self._split)

        # load the triviaqa data if needed
        if self._subset == "triviaqa":
            if config.triviaqa_path is not None:
                triviaqa_path = config.triviaqa_path
            else:
                triviaqa_path = FLEXRAG_CACHE_DIR / "datasets" / "trivia_qa"
            if not triviaqa_path.exists():
                snapshot_download(
                    repo_id="mandarjoshi/trivia_qa",
                    repo_type="dataset",
                    local_dir=triviaqa_path.as_posix(),
                    token=os.getenv("HF_TOKEN"),
                )
            triviaqa = load_dataset(
                triviaqa_path.as_posix(),
                name="unfiltered.nocontext",
                split=self._split,
            )

            def add_missing_data(x, trivia_qa_subset, triviaqa_map):
                i = triviaqa_map[x["id"]]
                x["input"] = trivia_qa_subset[i]["question"]
                x["original_answer"] = trivia_qa_subset[i]["answer"]["value"]
                return x

            triviaqa_map = dict(
                [(q_id, i) for i, q_id in enumerate(triviaqa["question_id"])]
            )
            subset = subset.filter(lambda x: x["id"] in triviaqa_map)
            subset = subset.map(
                add_missing_data,
                fn_kwargs=dict(trivia_qa_subset=triviaqa, triviaqa_map=triviaqa_map),
            )

        # prepare the data
        self._queries_data = {}
        self._answers_data = {}
        self._qrels_data = {}
        # save exact match positions for evaluation
        self._meta_data = {}
        for item in subset:
            self._queries_data[item["id"]] = item["input"]
            answers = []
            qrels = {}
            meta = {"provenance": []}
            for ans in item["output"]:
                # filter empty answers
                if ans["answer"] != "":
                    answers.append(ans["answer"])
                # the `start_paragraph_id` and `end_paragraph_id` are the same in
                # nq, hqa, and tqa, thus we use `start_paragraph_id` only
                for p in ans["provenance"]:
                    wikipedia_id = p["wikipedia_id"]
                    chunk_id = str(p["start_paragraph_id"])
                    context_id = f"{wikipedia_id}_{chunk_id}"
                    qrels[context_id] = 1.0
                    meta["provenance"].append(
                        {
                            "wikipedia_id": wikipedia_id,
                            "start_paragraph_id": p["start_paragraph_id"],
                            "end_paragraph_id": p["end_paragraph_id"],
                            "start_character": p["start_character"],
                            "end_character": p["end_character"],
                        }
                    )
            if "original_answer" in item:
                meta["original_answer"] = item["original_answer"]
                answers.append(item["original_answer"])
            self._answers_data[item["id"]] = list(set(answers))  # deduplicate answers
            self._qrels_data[item["id"]] = qrels
            self._meta_data[item["id"]] = meta
        return

    def _load_corpus(
        self, corpus_path: Path, load_meta: bool = False
    ) -> dict[str, Context]:
        logger.info("Loading KILT corpus...")
        wiki_data = {}
        reader = LineDelimitedReader(corpus_path)
        for document in reader:
            wikipedia_id = document["wikipedia_id"]
            meta_data = (
                {
                    "wikidata_info": document.get("wikidata_info", {}),
                    "categories": document.get("categories", []),
                    "history": document.get("history", {}),
                    "anchors": document.get("anchors", []),
                }
                if load_meta
                else {}
            )
            for chunk_id, chunk in enumerate(document["text"]):
                context_id = f"{wikipedia_id}_{chunk_id}"
                # PERFORMANCE NOTE: pydantic model instantiation is slow here
                context = Context(
                    context_id=context_id,
                    data={
                        "text": chunk,
                        "title": document["wikipedia_title"],
                    },
                    source="wikipedia",
                    meta_data=meta_data,
                )
                wiki_data[context_id] = context
        logger.info(f"Loaded {len(wiki_data)} contexts from KILT corpus.")
        return wiki_data

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
