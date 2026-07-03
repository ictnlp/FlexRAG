import json

import pytest

from flexrag.common import Context
from flexrag.datasets.corpora import CorpusView
from flexrag.datasets.corpora.corpus_dataset import _ContextMappingCorpus
from flexrag.datasets.corpora.wikipedia_attributeqa import (
    WikipediaAttributedQACorpus,
    WikipediaAttributedQACorpusConfig,
)


def _write_jsonl(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return


@pytest.fixture
def mapping_corpus() -> _ContextMappingCorpus:
    contexts = {
        "ctx-0": Context(context_id="ctx-0", data={"text": "zero"}),
        "ctx-1": Context(context_id="ctx-1", data={"text": "one"}),
        "ctx-2": Context(context_id="ctx-2", data={"text": "two"}),
        "ctx-3": Context(context_id="ctx-3", data={"text": "three"}),
    }
    return _ContextMappingCorpus(contexts)


class TestContextMappingCorpus:
    def test_int_indexing(self, mapping_corpus: _ContextMappingCorpus) -> None:
        assert mapping_corpus[0].context_id == "ctx-0"
        assert mapping_corpus[-1].context_id == "ctx-3"
        return

    def test_slice_returns_view(self, mapping_corpus: _ContextMappingCorpus) -> None:
        subset = mapping_corpus[1:3]
        assert isinstance(subset, CorpusView)
        assert len(subset) == 2
        assert [ctx.context_id for ctx in subset] == ["ctx-1", "ctx-2"]
        assert list(subset.context_ids) == ["ctx-1", "ctx-2"]
        assert list(subset.contexts.keys()) == ["ctx-1", "ctx-2"]
        assert subset[0].context_id == "ctx-1"
        assert subset[-1].context_id == "ctx-2"
        return

    def test_nested_and_reversed_slices(
        self, mapping_corpus: _ContextMappingCorpus
    ) -> None:
        reversed_subset = mapping_corpus[::-1]
        assert [ctx.context_id for ctx in reversed_subset] == [
            "ctx-3",
            "ctx-2",
            "ctx-1",
            "ctx-0",
        ]
        assert list(reversed_subset.contexts.keys()) == [
            "ctx-3",
            "ctx-2",
            "ctx-1",
            "ctx-0",
        ]

        nested_subset = mapping_corpus[1:][::2]
        assert isinstance(nested_subset, CorpusView)
        assert [ctx.context_id for ctx in nested_subset] == ["ctx-1", "ctx-3"]
        assert list(nested_subset.context_ids) == ["ctx-1", "ctx-3"]
        return

    @pytest.mark.parametrize("index", [4, -5])
    def test_out_of_range_index(
        self, mapping_corpus: _ContextMappingCorpus, index: int
    ) -> None:
        with pytest.raises(IndexError):
            _ = mapping_corpus[index]
        return

    @pytest.mark.parametrize("index", ["ctx-0", 1.5, None])
    def test_invalid_index_type(
        self, mapping_corpus: _ContextMappingCorpus, index: object
    ) -> None:
        with pytest.raises(TypeError):
            _ = mapping_corpus[index]
        return


class TestWikipediaAttributedQACorpusIndexing:
    def test_load_in_memory_required(self, tmp_path) -> None:
        wikipedia_dir = tmp_path / "wikipedia"
        wikipedia_dir.mkdir()
        _write_jsonl(wikipedia_dir / "b.jsonl", [{"id": "ctx-b", "text": "beta"}])

        corpus = WikipediaAttributedQACorpus(
            WikipediaAttributedQACorpusConfig(
                data_path=tmp_path.as_posix(),
                load_in_memory=False,
            )
        )

        with pytest.raises(
            RuntimeError,
            match="WikipediaAttributedQACorpus.__getitem__ requires load_in_memory=True.",
        ):
            _ = corpus[0]
        return

    def test_materialized_indexing_and_order(self, tmp_path) -> None:
        wikipedia_dir = tmp_path / "wikipedia"
        wikipedia_dir.mkdir()
        _write_jsonl(
            wikipedia_dir / "b.jsonl",
            [
                {"id": "ctx-b1", "text": "beta-1"},
                {"id": "ctx-b2", "text": "beta-2"},
            ],
        )
        _write_jsonl(
            wikipedia_dir / "a.jsonl",
            [{"id": "ctx-a1", "text": "alpha-1"}],
        )

        corpus = WikipediaAttributedQACorpus(
            WikipediaAttributedQACorpusConfig(
                data_path=tmp_path.as_posix(),
                load_in_memory=True,
            )
        )

        assert [ctx.context_id for ctx in corpus] == ["ctx-a1", "ctx-b1", "ctx-b2"]
        assert corpus[0].context_id == "ctx-a1"
        assert corpus[-1].context_id == "ctx-b2"

        subset = corpus[1:]
        assert isinstance(subset, CorpusView)
        assert [ctx.context_id for ctx in subset] == ["ctx-b1", "ctx-b2"]
        assert list(subset.contexts.keys()) == ["ctx-b1", "ctx-b2"]
        assert list(corpus.context_ids) == ["ctx-a1", "ctx-b1", "ctx-b2"]
        return
