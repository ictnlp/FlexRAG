import ast
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from datasets import load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, Context, configure

from ...core import IRQASample, MappingDataset
from ...corpora import WikipediaWikimediaCorpus, WikipediaWikimediaCorpusConfig
from ...corpora.corpus_dataset import _ContextMappingCorpus


@configure
class FramesDatasetConfig:
    """Configuration for FRAMES dataset.

    `FRAMES <https://arxiv.org/abs/2409.12941>`_ is a comprehensive evaluation
    benchmark for end-to-end RAG systems. Each question is paired with a gold
    answer and a set of supporting Wikipedia pages.

    :param data_path: The path to the FRAMES dataset repository. Default is None.
        If not provided, the dataset will be downloaded automatically.
    :type data_path: Optional[str]
    :param load_corpus: Whether to load the supporting Wikimedia corpus. Default is False.
    :type load_corpus: bool
    :param corpus_path: The path to the local Wikimedia corpus repository. Default is None.
        If not provided, the corpus will be loaded from the default cache directory.
    :type corpus_path: Optional[str]
    """

    data_path: Optional[str] = None
    load_corpus: bool = False
    corpus_path: Optional[str] = None


def _normalize_wiki_url(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path).rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _title_from_wiki_url(url: str) -> str:
    path = urlparse(url).path
    title = unquote(path.rsplit("/", 1)[-1])
    return title.replace("_", " ")


class FramesDataset(MappingDataset[IRQASample]):
    """Dataset for the FRAMES end-to-end RAG benchmark."""

    def __init__(self, config: FramesDatasetConfig):
        if config.data_path is not None:
            data_path = Path(config.data_path)
        else:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "frames"

        if not data_path.exists():
            snapshot_download(
                repo_id="google/frames-benchmark",
                repo_type="dataset",
                local_dir=data_path.as_posix(),
            )

        raw_dataset = load_dataset(data_path.as_posix(), split="test")

        self._queries_data: dict[str, str] = {}
        self._answers_data: dict[str, list[str]] = {}
        self._links_data: dict[str, list[str]] = {}
        self._metadata: dict[str, dict] = {}
        for item in raw_dataset:
            qid = str(item["Unnamed: 0"])
            wiki_links = ast.literal_eval(item["wiki_links"])
            wiki_links = [_normalize_wiki_url(link) for link in wiki_links]
            self._queries_data[qid] = item["Prompt"]
            self._answers_data[qid] = [item["Answer"]]
            self._links_data[qid] = wiki_links
            self._metadata[qid] = {
                "reasoning_types": item["reasoning_types"],
                "wiki_links": wiki_links,
            }
        self._qids = sorted(self._queries_data.keys(), key=int)

        self._context_data: dict[str, Context] | None = None
        if config.load_corpus:
            corpus = WikipediaWikimediaCorpus(
                WikipediaWikimediaCorpusConfig(
                    data_path=config.corpus_path,
                    subset="20231101.en",
                )
            )
            self._context_data = dict(corpus.contexts)
            self._context_by_url = {}
            self._context_by_title = {}
            for context in self._context_data.values():
                url = context.metadata.get("url", "")
                if url:
                    self._context_by_url[_normalize_wiki_url(url)] = context
                title = context.data.get("title", "")
                if title and title not in self._context_by_title:
                    self._context_by_title[title] = context
            self._corpus = _ContextMappingCorpus(self._context_data)
        else:
            self._context_by_url = {}
            self._context_by_title = {}
            self._corpus = None
        return

    def __len__(self) -> int:
        return len(self._qids)

    def _resolve_contexts(self, qid: str) -> list[Context]:
        contexts = []
        for link in self._links_data[qid]:
            context = self._context_by_url.get(link)
            if context is None:
                context = self._context_by_title.get(_title_from_wiki_url(link))
            if context is None:
                context = Context(
                    context_id=link,
                    data={"title": _title_from_wiki_url(link)},
                    source="wikimedia/wikipedia",
                    metadata={"url": link},
                )
            contexts.append(context)
        return contexts

    def get_item(self, index: int) -> IRQASample:
        qid = self._qids[index]
        contexts = self._resolve_contexts(qid)
        return IRQASample(
            question_id=qid,
            question=self._queries_data[qid],
            answers=self._answers_data[qid],
            contexts=contexts,
            qrels={
                ctx.context_id: 1.0 for ctx in contexts if ctx.context_id is not None
            },
            metadata=self._metadata[qid],
        )

    @property
    def corpus(self):
        return self._corpus
