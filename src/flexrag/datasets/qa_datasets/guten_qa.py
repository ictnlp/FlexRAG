from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
from huggingface_hub import hf_hub_download

from flexrag.common import FLEXRAG_CACHE_DIR, Choices, configure
from flexrag.common.dataclasses import Context

from .qa_dataset_base import KNOWLEDGE_QA_DATASETS, QA_DATASETS, KnowledgeQADatasetBase


@configure
class GutenQADatasetConfig:
    """Configuration for GutenQADataset.

    `GutenQA <https://arxiv.org/abs/2406.17526>`_ is a benchmark of 100
    public-domain narrative books with 3,000 expert-crafted question-answer pairs
    designed to evaluate retrieval performance.

    :param data_path: The path to the GutenQA dataset file. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :context_mode: How contexts are organized. Default is `lumber_chunk`.
        Available choices are:
            - `lumber_chunk`: Use pre-segmented chunks from LumberChunker.
            - `recursive_chunk`: Use recursively chunked text segments.
            - `semantic_chunk`: Use semantically chunked text segments.
            - `book`: Use entire books as contexts.
    :type context_mode: str
    """

    data_path: Optional[str] = None
    context_mode: Annotated[
        str,
        Choices(
            "lumber_chunk",
            "recursive_chunk",
            "semantic_chunk",
            "book",
        ),
    ] = "chunk"


@QA_DATASETS("guten_qa", config_class=GutenQADatasetConfig)
@KNOWLEDGE_QA_DATASETS("guten_qa", config_class=GutenQADatasetConfig)
class GutenQADataset(KnowledgeQADatasetBase):
    def __init__(self, config: GutenQADatasetConfig):
        self._context_mode = config.context_mode
        # prepare the data directory
        if config.data_path is None:
            data_dir = FLEXRAG_CACHE_DIR / "datasets" / "guten_qa"
        else:
            data_dir = Path(config.data_path)
        if not data_dir.exists():
            data_dir.parent.mkdir(parents=True, exist_ok=True)

        # download the dataset based on context_mode
        qa_path = data_dir / "questions.parquet"
        if not qa_path.exists():
            hf_hub_download(
                repo_id="LumberChunker/GutenQA",
                repo_type="dataset",
                filename="questions.parquet",
                local_dir=data_dir.as_posix(),
            )
        match self._context_mode:
            case "lumber_chunk":
                corpus_path = data_dir / "gutenqa_chunks.parquet"
                if not corpus_path.exists():
                    hf_hub_download(
                        repo_id="LumberChunker/GutenQA",
                        repo_type="dataset",
                        filename="gutenqa_chunks.parquet",
                        local_dir=data_dir.as_posix(),
                    )
            case "recursive_chunk":
                corpus_path = data_dir / "GutenQA_recursive.parquet"
                if not corpus_path.exists():
                    hf_hub_download(
                        repo_id="LumberChunker/GutenQA_Recursive",
                        repo_type="dataset",
                        filename="GutenQA_recursive.parquet",
                        local_dir=data_dir.as_posix(),
                    )
            case "semantic_chunk":
                corpus_path = data_dir / "GutenQA_semantic.parquet"
                if not corpus_path.exists():
                    hf_hub_download(
                        repo_id="LumberChunker/GutenQA_Semantic",
                        repo_type="dataset",
                        filename="GutenQA_semantic.parquet",
                        local_dir=data_dir.as_posix(),
                    )
            case "book":
                corpus_path = data_dir / "GutenQA_paragraphs.parquet"
                if not corpus_path.exists():
                    hf_hub_download(
                        repo_id="LumberChunker/GutenQA_Paragraphs",
                        repo_type="dataset",
                        filename="GutenQA_paragraphs.parquet",
                        local_dir=data_dir.as_posix(),
                    )
            case _:
                raise ValueError(
                    f"Invalid context_mode: {self._context_mode}. "
                    "Choose from 'lumber_chunk', 'recursive_chunk', "
                    "'semantic_chunk', 'propositional_chunk', 'book'."
                )

        # load the corpus
        self._context_data = {}
        corpus = pd.read_parquet(corpus_path)
        if self._context_mode == "book":
            for _, group in corpus.groupby("Book ID"):
                book_id = str(group.iloc[0]["Book ID"])
                context = Context(
                    context_id=book_id,
                    data={
                        "Book Name": group.iloc[0]["Book Name"],
                        "text": "\n".join(group["Chunk"].tolist()),
                    },
                    source="Gutenberg",
                )
                self._context_data[book_id] = context
        else:
            for _, row in corpus.iterrows():
                ctx_id = f"{row['Book ID']}_{row['Chunk ID']}"
                context = Context(
                    context_id=ctx_id,
                    data={
                        "Book Name": row["Book Name"],
                        "Chapter": row.get("Chapter", ""),
                        "text": row["Chunk"],
                    },
                    source="Gutenberg",
                )
                self._context_data[ctx_id] = context

        # load the QA pairs
        self._queries_data = {}
        self._answers_data = {}
        self._qrels_data = {}
        qa_pairs = pd.read_parquet(qa_path)
        for idx, row in qa_pairs.iterrows():
            if self._context_mode == "book":
                ctx_id = str(row["Book ID"])
            else:
                ctx_id = f"{row['Book ID']}_{row['Chunk ID']}"
            query_id = str(idx)
            self._queries_data[query_id] = row["Question"]
            self._answers_data[query_id] = [row["Answer"]]
            self._qrels_data[query_id] = {ctx_id: 1.0}
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
