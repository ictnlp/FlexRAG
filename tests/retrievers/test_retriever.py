import asyncio
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from flexrag.common import Context
from flexrag.datasets.reader import LineDelimitedReader
from flexrag.retrievers import (
    ElasticRetriever,
    ElasticRetrieverConfig,
    FlexRetriever,
    FlexRetrieverConfig,
    RetrieverBase,
)
from flexrag.retrievers.index import (
    RETRIEVER_INDEX,
    BM25IndexConfig,
    RetrieverIndexConfig,
)

TEST_CORPUS_PATH = (
    Path(__file__).resolve().parents[1] / "support" / "data" / "testcorp.jsonl"
)


def load_test_corpus_slice(path: Path, start: int, stop: int) -> list[Context]:
    contexts = []
    reader = LineDelimitedReader(path)
    for idx, data in enumerate(reader):
        if idx < start:
            continue
        if idx >= stop:
            break
        payload = dict(data)
        context_id = payload.pop("id")
        contexts.append(Context(context_id=context_id, data=payload))
    return contexts


def build_bm25_index():
    return RETRIEVER_INDEX.load(
        RetrieverIndexConfig(
            index_type="bm25",
            bm25_config=BM25IndexConfig(
                indexed_fields=["title", "section", "text"],
                merge_method="max",
                show_progress=False,
            ),
        )
    )


def build_text_bm25_index():
    return RETRIEVER_INDEX.load(
        RetrieverIndexConfig(
            index_type="bm25",
            bm25_config=BM25IndexConfig(
                indexed_fields=["text"],
                show_progress=False,
            ),
        )
    )


class FakeAddableIndex:
    is_addable = True
    infimum = 0.0
    supremum = 1.0

    def __init__(self) -> None:
        self.context_ids: list[str] = []
        self.inserted_batches: list[list[str]] = []

    def build_index(self, context_ids, data, batch_size=32, scratch_path=None) -> None:
        self.context_ids = list(context_ids)
        list(data)
        return

    def insert_batch(
        self,
        context_ids,
        data,
        batch_size=32,
        log_interval=10000,
        display="auto",
    ) -> None:
        ids = list(context_ids)
        self.inserted_batches.append(ids)
        self.context_ids.extend(ids)
        list(data)
        return

    def search(self, query, top_k, **search_kwargs):
        hits = self.context_ids[:top_k]
        context_ids = [hits for _ in query]
        scores = np.tile(
            np.arange(len(hits), 0, -1, dtype=float),
            (len(query), 1),
        )
        return context_ids, scores

    def save_to_local(self, index_path: str) -> None:
        os.makedirs(index_path, exist_ok=True)
        Path(index_path, "fake.index").write_text("fake", encoding="utf-8")
        return

    def clear(self) -> None:
        self.context_ids = []
        return

    def __len__(self) -> int:
        return len(self.context_ids)


class FakeScratchIndex(FakeAddableIndex):
    def __init__(self) -> None:
        super().__init__()
        self.scratch_path: str | None = None

    def build_index(self, context_ids, data, batch_size=32, scratch_path=None) -> None:
        self.scratch_path = scratch_path
        assert scratch_path is not None
        Path(scratch_path, "embedding.npy").write_text("scratch", encoding="utf-8")
        super().build_index(context_ids, data, batch_size=batch_size)
        return


class FakeRepoUrl:
    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id

    def __str__(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"


class FakeHfApi:
    snapshot_path: str
    upload_calls: list[dict] = []
    upload_file_calls: list[dict] = []

    def __init__(self, token=None) -> None:
        self.token = token

    def repo_info(self, repo_id: str):
        return type("RepoInfo", (), {"id": repo_id})()

    def snapshot_download(self, **kwargs):
        return self.snapshot_path

    def create_repo(
        self,
        repo_id,
        token=None,
        private=False,
        repo_type=None,
        exist_ok=True,
    ):
        return FakeRepoUrl(repo_id)

    def upload_folder(self, **kwargs):
        self.upload_calls.append(kwargs)
        return

    def upload_file(self, **kwargs):
        self.upload_file_calls.append(kwargs)
        return


class TestRetrievers:
    query = [
        "Who is Bruce Wayne?",
        "What is the capital of China?",
    ]

    def run_retriever(self, retriever: RetrieverBase):
        retriever.clear()
        assert len(retriever) == 0

        data_path = TEST_CORPUS_PATH
        dataset1 = load_test_corpus_slice(data_path, 0, 10000)
        dataset2 = load_test_corpus_slice(data_path, 10000, 20000)

        retriever.add_passages(dataset1)
        assert len(retriever) == 10000

        ctxs = retriever.search(self.query, disable_cache=True)
        assert len(ctxs) == 2
        assert len(ctxs[0]) == 10

        ctxs = retriever.search(self.query, disable_cache=True, top_k=5)
        assert len(ctxs) == 2
        assert len(ctxs[0]) == 5

        retriever.add_passages(dataset2)
        assert len(retriever) == 20000

        ctxs = retriever.search(self.query, disable_cache=True)
        assert len(ctxs) == 2
        assert len(ctxs[0]) == 10

        ctxs = retriever.search(self.query, disable_cache=True, top_k=5)
        assert len(ctxs) == 2
        assert len(ctxs[0]) == 5

        retriever.clear()
        assert len(retriever) == 0
        return

    def test_flex_retriever(self):
        data_path = TEST_CORPUS_PATH
        dataset1 = load_test_corpus_slice(data_path, 0, 50)
        dataset2 = load_test_corpus_slice(data_path, 50, 60)

        with tempfile.TemporaryDirectory() as tempdir:
            retriever = FlexRetriever(FlexRetrieverConfig(batch_size=16), tempdir)
            assert Path(tempdir, "metadata.json").exists()
            assert Path(tempdir, "database.lmdb").exists()
            assert Path(tempdir, "indexes").exists()

            retriever.add_passages(dataset1, log_interval=1000)
            assert len(retriever) == 50
            assert retriever.count() == 50
            assert retriever[dataset1[0].context_id].data == dataset1[0].data

            retriever.add_index("bm25", build_bm25_index())
            assert Path(tempdir, "indexes", "bm25", "context_mapping.pkl").exists()
            assert Path(tempdir, "indexes", "bm25", "raw").exists()

            ctxs = retriever.search(
                self.query,
                disable_cache=True,
                top_k=5,
                used_indexes=["bm25"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5
            assert len(ctxs[1]) == 5

            retriever.add_passages(dataset2, log_interval=1000)
            assert len(retriever) == 60
            assert "bm25" in retriever.state.dirty_indexes
            with pytest.raises(RuntimeError, match="dirty indexes"):
                retriever.search(
                    self.query,
                    disable_cache=True,
                    top_k=5,
                    used_indexes=["bm25"],
                )

            retriever.rebuild_index("bm25")
            assert "bm25" not in retriever.state.dirty_indexes
            ctxs = retriever.search(
                self.query,
                disable_cache=True,
                top_k=5,
                used_indexes=["bm25"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5

            retriever.close()
            reopened = FlexRetriever(FlexRetrieverConfig(batch_size=16), tempdir)
            assert len(reopened) == 60
            assert "bm25" in reopened.index_table
            ctxs = reopened.search(
                self.query,
                disable_cache=True,
                top_k=5,
                used_indexes=["bm25"],
            )
            assert len(ctxs) == 2
            assert len(ctxs[0]) == 5

            with tempfile.TemporaryDirectory() as export_parent:
                export_path = Path(export_parent, "exported-flex")
                reopened.export_to(export_path)
                exported = FlexRetriever(
                    FlexRetrieverConfig(batch_size=16),
                    export_path,
                )
                assert len(exported) == 60
                assert "bm25" in exported.index_table
                exported.close()
            reopened.close()
        return

    def test_flex_retriever_path_semantics(self):
        with tempfile.TemporaryDirectory() as tempdir:
            collection_path = Path(tempdir, "collection")
            retriever = FlexRetriever(FlexRetrieverConfig(), collection_path)
            assert Path(collection_path, "metadata.json").exists()
            retriever.close()

            non_collection = Path(tempdir, "non-collection")
            non_collection.mkdir()
            Path(non_collection, "random.txt").write_text("not a collection")
            with pytest.raises(FileExistsError):
                FlexRetriever(FlexRetrieverConfig(), non_collection)

            file_path = Path(tempdir, "file")
            file_path.write_text("not a collection")
            with pytest.raises(NotADirectoryError):
                FlexRetriever(FlexRetrieverConfig(), file_path)
        return

    def test_flex_retriever_duplicate_context_id(self):
        with tempfile.TemporaryDirectory() as tempdir:
            retriever = FlexRetriever(FlexRetrieverConfig(batch_size=2), tempdir)
            retriever.add_passages([Context(context_id="a", data={"text": "alpha"})])
            with pytest.raises(ValueError, match="Duplicate context_id"):
                retriever.add_passages([Context(context_id="a", data={"text": "beta"})])
            with pytest.raises(ValueError, match="Duplicate context_id"):
                retriever.add_passages(
                    [
                        Context(context_id="b", data={"text": "beta"}),
                        Context(context_id="b", data={"text": "beta again"}),
                    ]
                )
            with pytest.raises(ValueError, match="context_id is required"):
                retriever.add_passages([Context(data={"text": "missing id"})])
            retriever.close()
        return

    def test_flex_retriever_rebuilds_dirty_index(self):
        with tempfile.TemporaryDirectory() as tempdir:
            retriever = FlexRetriever(FlexRetrieverConfig(batch_size=2), tempdir)
            retriever.add_index("bm25", build_text_bm25_index())
            retriever.add_passages(
                [
                    Context(
                        context_id="bruce",
                        data={"text": "Bruce Wayne guards Gotham."},
                    )
                ]
            )
            assert "bm25" in retriever.state.dirty_indexes
            retriever.rebuild_index("bm25")
            retriever.add_passages(
                [
                    Context(
                        context_id="capital",
                        data={"text": "Beijing is the capital of China."},
                    )
                ]
            )

            with pytest.raises(RuntimeError, match="dirty indexes"):
                retriever.search(
                    ["Bruce Wayne", "capital of China"],
                    disable_cache=True,
                    top_k=1,
                    used_indexes=["bm25"],
                )

            retriever.rebuild_index()
            results = retriever.search(
                ["Bruce Wayne", "capital of China"],
                disable_cache=True,
                top_k=1,
                used_indexes=["bm25"],
            )

            assert results[0][0].context_id == "bruce"
            assert results[1][0].context_id == "capital"
            retriever.close()

    def test_flex_retriever_updates_addable_index_incrementally(self):
        with tempfile.TemporaryDirectory() as tempdir:
            retriever = FlexRetriever(FlexRetrieverConfig(batch_size=2), tempdir)
            retriever.add_passages(
                [
                    Context(context_id="a", data={"text": "alpha"}),
                    Context(context_id="b", data={"text": "beta"}),
                ]
            )
            index = FakeAddableIndex()
            retriever.add_index("fake", index)
            retriever.add_passages([Context(context_id="c", data={"text": "gamma"})])

            assert index.inserted_batches == [["c"]]
            assert "fake" not in retriever.state.dirty_indexes
            results = retriever.search(
                "anything",
                disable_cache=True,
                top_k=3,
                used_indexes=["fake"],
            )
            assert [ctx.context_id for ctx in results[0]] == ["a", "b", "c"]
            retriever.close()

    def test_flex_retriever_uses_collection_local_scratch_path(self):
        with tempfile.TemporaryDirectory() as tempdir:
            retriever = FlexRetriever(FlexRetrieverConfig(batch_size=2), tempdir)
            retriever.add_passages([Context(context_id="a", data={"text": "alpha"})])
            index = FakeScratchIndex()
            retriever.add_index("fake", index)

            assert index.scratch_path == str(
                Path(tempdir, "indexes", ".scratch", "fake")
            )
            assert not Path(index.scratch_path).exists()
            retriever.close()

    def test_flex_retriever_hub_helpers(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tempdir:
            retriever = FlexRetriever(FlexRetrieverConfig(batch_size=2), tempdir)
            retriever.add_passages([Context(context_id="a", data={"text": "alpha"})])
            retriever.add_index("bm25", build_text_bm25_index())
            retriever.add_passages([Context(context_id="b", data={"text": "beta"})])

            FakeHfApi.snapshot_path = tempdir
            FakeHfApi.upload_calls = []
            FakeHfApi.upload_file_calls = []
            monkeypatch.setattr("flexrag.retrievers.flex_retriever.HfApi", FakeHfApi)

            with pytest.raises(RuntimeError, match="dirty indexes"):
                retriever.push_to_hub("org/repo")
            url = retriever.push_to_hub("org/repo", allow_dirty=True)
            assert url == "https://huggingface.co/org/repo"
            assert FakeHfApi.upload_calls[-1]["folder_path"] == tempdir
            assert FakeHfApi.upload_file_calls[-1]["path_in_repo"] == "README.md"
            card = FakeHfApi.upload_file_calls[-1]["path_or_fileobj"].decode("utf-8")
            assert "FlexRetriever.from_hub" in card
            assert "Context count: `2`" in card
            assert "- `bm25`" in card
            assert "Dirty Indexes" in card
            assert not Path(tempdir, "README.md").exists()

            retriever.close()
            loaded = FlexRetriever.from_hub(
                "org/repo",
                cfg=FlexRetrieverConfig(batch_size=2),
            )
            assert len(loaded) == 2
            loaded.close()

    def test_elastic_retriever(self, mock_es_client):
        retriever = ElasticRetriever(
            ElasticRetrieverConfig(
                host="http://127.0.0.1:9200",
                index_name="testing",
            )
        )
        self.run_retriever(retriever)

        data_path = TEST_CORPUS_PATH
        dataset = load_test_corpus_slice(data_path, 0, 20)
        asyncio.run(retriever.async_add_passages(dataset))
        assert asyncio.run(retriever.async_count()) == 20
        ctxs = asyncio.run(
            retriever.async_search(self.query, disable_cache=True, top_k=5)
        )
        assert len(ctxs) == 2
        assert len(ctxs[0]) == 5
        asyncio.run(retriever.async_clear())
        assert asyncio.run(retriever.async_count()) == 0
        asyncio.run(retriever.aclose())
        return
