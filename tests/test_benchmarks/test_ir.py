from collections.abc import Iterable, Mapping

import pytest

from flexrag.common import Context
from flexrag.datasets.benchmarks import (
    KiltDataset,
    KiltDatasetConfig,
    MSMARCODataset,
    MSMARCODatasetConfig,
    MTEBDataset,
    MTEBDatasetConfig,
    MultiLongDocRetrievalDataset,
    MultiLongDocRetrievalDatasetConfig,
)
from flexrag.datasets.core import IRSample


class TestMSMARCODataset:
    def valid_ir_sample(self, item):
        assert isinstance(item, IRSample)
        assert item.question_id is not None
        assert len(item.contexts) > 0
        assert isinstance(item.contexts[0], Context)
        return

    @pytest.mark.parametrize(
        "split",
        [
            "train",
            "dev",
            "trec-dl-2019",
            "trec-dl-2020",
            "trec-dl-hard",
            "orcas",
        ],
    )
    def test_document_ranking_v1(self, split):
        dataset = MSMARCODataset(
            MSMARCODatasetConfig(
                subset="msmarco_document_ranking_v1",
                split=split,
                data_path=None,
                load_corpus=False,
            )
        )
        for item in dataset:
            self.valid_ir_sample(item)
        assert isinstance(dataset.contexts, Mapping)
        assert isinstance(dataset.qrels, Mapping)
        assert isinstance(dataset.context_ids, Iterable)
        print(f"MSMARCO Document Ranking V1 {split} split passed.")
        print(f"Number of samples: {len(dataset)}")
        return

    @pytest.mark.parametrize(
        "split",
        [
            "train",
            "dev1",
            "dev2",
            "trec-dl-2019",
            "trec-dl-2020",
            "trec-dl-2021",
            "trec-dl-2022",
        ],
    )
    def test_document_ranking_v2(self, split):
        dataset = MSMARCODataset(
            MSMARCODatasetConfig(
                subset="msmarco_document_ranking_v2",
                split=split,
                data_path=None,
                load_corpus=False,
            )
        )
        for item in dataset:
            self.valid_ir_sample(item)
        assert isinstance(dataset.contexts, Mapping)
        assert isinstance(dataset.qrels, Mapping)
        assert isinstance(dataset.context_ids, Iterable)
        print(f"MSMARCO Document Ranking V2 {split} split passed.")
        print(f"Number of samples: {len(dataset)}")
        return

    @pytest.mark.parametrize(
        "split",
        [
            "train",
            "dev",
            "trec-dl-2019",
            "trec-dl-2020",
            "trec-dl-hard",
        ],
    )
    def test_passage_ranking_v1(self, split):
        dataset = MSMARCODataset(
            MSMARCODatasetConfig(
                subset="msmarco_passage_ranking_v1",
                split=split,
                data_path=None,
                load_corpus=False,
            )
        )
        for item in dataset:
            self.valid_ir_sample(item)
        assert isinstance(dataset.contexts, Mapping)
        assert isinstance(dataset.qrels, Mapping)
        assert isinstance(dataset.context_ids, Iterable)
        print(f"MSMARCO Passage Ranking V1 {split} split passed.")
        print(f"Number of samples: {len(dataset)}")
        return

    @pytest.mark.parametrize(
        "split",
        [
            "train",
            "dev1",
            "dev2",
            "trec-dl-2021",
            "trec-dl-2022",
        ],
    )
    def test_passage_ranking_v2(self, split):
        dataset = MSMARCODataset(
            MSMARCODatasetConfig(
                subset="msmarco_passage_ranking_v2",
                split=split,
                data_path=None,
                load_corpus=False,
            )
        )
        for item in dataset:
            self.valid_ir_sample(item)
        assert isinstance(dataset.contexts, Mapping)
        assert isinstance(dataset.qrels, Mapping)
        assert isinstance(dataset.context_ids, Iterable)
        print(f"MSMARCO Passage Ranking V2 {split} split passed.")
        print(f"Number of samples: {len(dataset)}")
        return


class TestMultiLongDocRetrievalDataset:
    def valid_ir_sample(self, item, allow_empty_contexts=False):
        assert isinstance(item, IRSample)
        assert item.question_id is not None
        if not allow_empty_contexts:
            assert len(item.contexts) > 0
        if len(item.contexts) > 0:
            assert isinstance(item.contexts[0], Context)
        return

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    @pytest.mark.parametrize(
        "lang",
        [
            "ar",
            "de",
            "en",
            "es",
            "fr",
            "hi",
            "it",
            "ja",
            "ko",
            "pt",
            "ru",
            "th",
            "zh",
        ],
    )
    def test_mldr(self, split, lang):
        dataset = MultiLongDocRetrievalDataset(
            MultiLongDocRetrievalDatasetConfig(split=split, lang=lang)
        )
        for item in dataset:
            self.valid_ir_sample(item, True)
        assert isinstance(dataset.contexts, Mapping)
        assert isinstance(dataset.qrels, Mapping)
        assert isinstance(dataset.context_ids, Iterable)
        print(f"MLDR-{split}-{lang} dataset length: {len(dataset)}")
        print(f"MLDR-{split}-{lang} dataset test passed.")
        return


class TestMTEBDataset:
    def valid_ir_sample(self, item):
        assert isinstance(item, IRSample)
        assert item.question_id is not None
        return

    @pytest.mark.parametrize(
        "subset",
        ["nq", "trec-covid", "fiqa", "hotpotqa", "scifact", "nfcorpus"],
    )
    def test_traditional_datasets(self, subset):
        dataset = MTEBDataset(
            MTEBDatasetConfig(
                subset=subset,
                split="test",
                load_corpus=True,
            )
        )
        for item in dataset:
            self.valid_ir_sample(item)
        assert isinstance(dataset.contexts, Mapping)
        assert isinstance(dataset.qrels, Mapping)
        assert isinstance(dataset.context_ids, Iterable)
        print(f"MTEB-{subset}-test dataset length: {len(dataset)}")
        print(f"MTEB-{subset}-test dataset test passed.")
        return

    @pytest.mark.parametrize(
        "subset_split",
        [
            "LEMBWikimQARetrieval_test",
            "LEMBSummScreenFDRetrieval_validation",
            "LEMBQMSumRetrieval_test",
        ],
    )
    def test_lemb_datasets(self, subset_split):
        subset, split = subset_split.split("_")
        dataset = MTEBDataset(
            MTEBDatasetConfig(
                subset=subset,
                split=split,
                load_corpus=True,
            )
        )
        for item in dataset:
            self.valid_ir_sample(item)
        assert isinstance(dataset.contexts, Mapping)
        assert isinstance(dataset.qrels, Mapping)
        assert isinstance(dataset.context_ids, Iterable)
        print(f"MTEB-{subset}-{split} dataset length: {len(dataset)}")
        print(f"MTEB-{subset}-{split} dataset test passed.")
        return


class TestKILTDataset:
    @pytest.mark.parametrize(
        "subset",
        [
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
        ],
    )
    @pytest.mark.parametrize("split", ["validation", "test"])
    def test_kilt(self, subset, split):
        dataset = KiltDataset(
            KiltDatasetConfig(subset=subset, split=split, load_corpus=False)
        )
        for item in dataset:
            if split == "validation":
                if hasattr(item, "answers"):
                    assert len(item.answers) > 0
                else:
                    assert len(item.golden_responses) > 0
                assert len(item.contexts) > 0
                assert isinstance(item.contexts[0], Context)
            pass
        print(f"KILT-{subset}-{split} dataset length: {len(dataset)}")
        print(f"KILT-{subset}-{split} dataset test passed.")
        return
