import os
from pathlib import Path
from typing import Annotated, Optional

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, configure
from flexrag.common.dataclasses import ChatMessages, ChatTurn, Context
from flexrag.common.logging import LOGGER_MANAGER
from flexrag.common.misc import download

from ...core import DATASETS, IRDialogueSample, IRMCSample, IRQASample, MappingDataset
from ...corpora.corpus_dataset import _ContextMappingCorpus
from ...reader import LineDelimitedReader

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.kilt_qa")


@configure
class KiltDatasetConfig:
    """Configuration for KiltDataset.

    `KILT <https://arxiv.org/abs/2009.02252>`_ is a unified evaluation framework
    for knowledge-intensive language tasks that grounds multiple tasks in a shared
    Wikipedia snapshot, enabling models to access external knowledge efficiently
    while being evaluated on both task performance and evidence provenance.

    For QA tasks, Entity Linking tasks, and Slot Filling tasks, the dataset
    provides IRQASample as the data sample type.
    For Dialogue tasks, the dataset provides IRDialogueSample as the data
    sample type.
    For Fact Checking tasks, the dataset provides IRMCSample as the data
    sample type.

    :param data_path: The path to the KILT dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param subset: The subset of KILT to use. Default is `nq`.
        Available choices are:

        - QA Tasks: hotpotqa (HotpotQA), nq (Natural Questions), triviaqa (TriviaQA), eli5 (ELI5)
        - Fact Checking Task: fever (FEVER)
        - Entity Linking Tasks: aidayago2 (AIDA-CoNLL-YAGO), wned (WNED-WIKI), cweb (WNED-CWEB)
        - Slot Filling Tasks: trex (T-REx), zsre (Zero Shot RE)
        - Dialogue Task: wow (Wizard of Wikipedia)
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
        If not provided, the corpus will be downloaded automatically.
    :type corpus_path: Optional[str]
    :param triviaqa_path: The path to the TriviaQA dataset file. Default is None.
        As the KILT release does not include the full TriviaQA dataset, the original
        TriviaQA dataset needs to be downloaded separately. If not provided, the
        dataset will be downloaded automatically to the cache directory.
    :type triviaqa_path: Optional[str]
    :param use_full_wiki: Whether to use the full Wikipedia pages instead of the
        chunked version. Default is False.
    :type use_full_wiki: bool
    """

    data_path: Optional[str] = None
    subset: Annotated[
        str,
        Choices(
            "hotpotqa",
            "nq",
            "triviaqa",
            "eli5",
            "fever",
            "aidayago2",
            "wned",
            "cweb",
            "trex",
            "zsre",
            "wow",
        ),
    ] = "nq"
    split: Annotated[str, Choices("train", "validation", "test")] = "validation"
    load_corpus: bool = True
    corpus_path: Optional[str] = None
    triviaqa_path: Optional[str] = None
    use_full_wiki: bool = False


CORPUS_URL = "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"


@DATASETS("kilt", config_class=KiltDatasetConfig)
class KiltDataset(MappingDataset[IRQASample | IRDialogueSample | IRMCSample]):
    def __init__(self, config: KiltDatasetConfig):
        self._subset = config.subset
        self._split = config.split
        self._full_wiki = config.use_full_wiki
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
                corpus_path = (
                    FLEXRAG_CACHE_DIR
                    / "corpora"
                    / "enwiki_2019_kilt"
                    / "kilt_knowledgesource.json"
                )
            if not corpus_path.exists():
                download(CORPUS_URL, corpus_path)
            self._context_data = self._load_corpus(
                corpus_path, full_wiki=self._full_wiki
            )
            self._corpus = _ContextMappingCorpus(self._context_data)
        else:
            self._context_data = None
            self._corpus = None

        # load the QA pairs
        if self._subset == "triviaqa":
            data_name = "triviaqa_support_only"
        elif self._subset == "zsre":
            data_name = "structured_zeroshot"
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
            meta = item.get("meta_data", {})
            meta["provenance"] = []
            for ans in item["output"]:
                # filter empty answers
                if ans["answer"] != "":
                    answers.append(ans["answer"])
                # parse provenance
                for p in ans["provenance"]:
                    wikipedia_id = p["wikipedia_id"]
                    chunk_ids = []
                    for cid in range(
                        p["start_paragraph_id"], p["end_paragraph_id"] + 1
                    ):
                        chunk_ids.append(str(cid))
                    if self._full_wiki:
                        qrels[wikipedia_id] = 1.0
                    else:
                        for chunk_id in chunk_ids:
                            qrels[f"{wikipedia_id}_{chunk_id}"] = 1.0
                    # save original provenance info
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
        self, corpus_path: Path, load_meta: bool = False, full_wiki: bool = False
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
            if full_wiki:
                context = Context(
                    context_id=wikipedia_id,
                    data={
                        "text": "".join(document["text"]),
                        "title": document["wikipedia_title"],
                    },
                    source="wikipedia",
                    meta_data=meta_data,
                )
                wiki_data[wikipedia_id] = context
            else:
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

    def __len__(self) -> int:
        return len(self._queries_data)

    def get_item(self, index: int) -> IRQASample | IRDialogueSample | IRMCSample:
        qid = list(self._queries_data.keys())[index]
        ctx_ids = list(self._qrels_data[qid].keys())
        # prepare contexts
        if self._context_data is not None:
            contexts = [self._context_data[ctx_id] for ctx_id in ctx_ids]
        else:
            contexts = [
                Context(context_id=ctx_id, data={}, source="wikipedia", meta_data={})
                for ctx_id in ctx_ids
            ]

        # prepare sample for fact checking task
        if self._subset in {"fever"}:
            choices = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
            if self._answers_data[qid]:
                answers = [choices.index(self._answers_data[qid][0])]
            else:
                answers = None
            sample = IRMCSample(
                question_id=qid,
                question=self._queries_data[qid],
                choices=choices,
                answers=answers,
                contexts=contexts,
                meta_data=self._meta_data[qid],
            )
        # prepare sample for dialogue task
        elif self._subset in {"wow"}:
            messages = []
            for n, turn in enumerate(self._queries_data[qid].split("\n")):
                role = "user" if n % 2 == 0 else "assistant"
                messages.append({"role": role, "content": turn})
            messages = ChatMessages.from_list(messages)
            if self._answers_data[qid]:
                responses = [
                    ChatTurn(
                        role="assistant",
                        content=self._answers_data[qid][0],
                    )
                ]
            else:
                responses = []
            sample = IRDialogueSample(
                question_id=qid,
                messages=messages,
                golden_responses=responses,
                contexts=contexts,
                meta_data=self._meta_data[qid],
            )
        # prepare sample for other tasks
        else:
            answers = self._answers_data[qid] if self._answers_data[qid] else None
            sample = IRQASample(
                question_id=qid,
                question=self._queries_data[qid],
                answers=answers,
                contexts=contexts,
                meta_data=self._meta_data[qid],
            )
        return sample

    @property
    def corpus(self):
        return self._corpus
