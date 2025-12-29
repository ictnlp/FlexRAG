import pytest

from flexrag.common import LOGGER_MANAGER, Context
from flexrag.datasets.benchmarks import (
    BrowseCompDataset,
    BrowseCompDatasetConfig,
    CRUDQADataset,
    CRUDQADatasetConfig,
    DeepSearchQADataset,
    DeepSearchQADatasetConfig,
    GAIADataset,
    GAIADatasetConfig,
    GutenQADataset,
    GutenQADatasetConfig,
    KiltDataset,
    KiltDatasetConfig,
    LongBenchDataset,
    LongBenchDatasetConfig,
    LongBenchV2Dataset,
    LongBenchV2DatasetConfig,
    MSMARCODataset,
    MSMARCODatasetConfig,
    MultihopRAGDataset,
    MultihopRAGDatasetConfig,
    MultiLongDocRetrievalDataset,
    MultiLongDocRetrievalDatasetConfig,
    NarrativeQADataset,
    NarrativeQADatasetConfig,
    NovelQAConfig,
    NovelQADataset,
    QuALITYDataset,
    QuALITYDatasetConfig,
    SimpleQADataset,
    SimpleQADatasetConfig,
    SQuADDataset,
    SQuADDatasetConfig,
)
from flexrag.datasets.core import (
    ContextualMCSample,
    ContextualQASample,
    IRSample,
    QASample,
)

logger = LOGGER_MANAGER.get_logger("tests.datasets")


class TestRAGEvalDataset:

    def test_deepsearch_qa(self):
        dataset = DeepSearchQADataset(DeepSearchQADatasetConfig())
        for item in dataset:
            assert isinstance(item, QASample)
        print(f"DeepSearch QA dataset length: {len(dataset)}")
        print("DeepSearch QA dataset test passed.")
        return

    def test_browsecomp(self):
        dataset = BrowseCompDataset(BrowseCompDatasetConfig())
        for item in dataset:
            assert isinstance(item, QASample)
        print(f"BrowseComp dataset length: {len(dataset)}")
        print("BrowseComp dataset test passed.")
        return

    @pytest.mark.parametrize(
        "subset",
        [
            "2023_all",
            "2023_level1",
            "2023_level2",
            "2023_level3",
        ],
    )
    @pytest.mark.parametrize("split", ["validation", "test"])
    def test_gaia(self, subset, split):
        dataset = GAIADataset(GAIADatasetConfig())
        for item in dataset:
            assert isinstance(item, QASample)
        print(f"GAIA-{subset}-{split} dataset length: {len(dataset)}")
        print(f"GAIA-{subset}-{split} dataset test passed.")
        return

    @pytest.mark.parametrize(
        "subset",
        [
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
        ],
    )
    def test_longbench(self, subset):
        dataset = LongBenchDataset(LongBenchDatasetConfig(subset=subset))
        for item in dataset:
            assert isinstance(item, ContextualQASample)
        print(f"LongBench-{subset} dataset length: {len(dataset)}")
        print(f"LongBench-{subset} dataset test passed.")
        return

    def test_longbench_v2(self):
        dataset = LongBenchV2Dataset(LongBenchV2DatasetConfig())
        for item in dataset:
            assert isinstance(item, ContextualMCSample)
        print(f"LongBenchV2 dataset length: {len(dataset)}")
        print("LongBenchV2 dataset test passed.")
        return

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_quality(self, split):
        dataset = QuALITYDataset(QuALITYDatasetConfig(split=split))
        for item in dataset:
            assert isinstance(item, ContextualMCSample)
        print(f"QuALITY-{split} dataset length: {len(dataset)}")
        print(f"QuALITY-{split} dataset test passed.")
        return

    def test_novel_qa(self):
        dataset = NovelQADataset(NovelQAConfig())
        for item in dataset:
            assert isinstance(item, ContextualMCSample)
        print(f"NovelQA dataset length: {len(dataset)}")
        print("NovelQA dataset test passed.")
        return

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

    @pytest.mark.parametrize("ctx_mode", ["lumber_chunk", "book"])
    def test_guten_qa(self, ctx_mode):
        dataset = GutenQADataset(GutenQADatasetConfig(context_mode=ctx_mode))
        for item in dataset:
            assert isinstance(item, ContextualQASample)
            assert len(item.contexts) > 0
            assert isinstance(item.contexts[0], Context)
        print(f"GutenQA-{ctx_mode} dataset length: {len(dataset)}")
        print(f"GutenQA-{ctx_mode} dataset test passed.")
        return

    @pytest.mark.parametrize("split", ["train", "validation", "test"])
    def test_narrative_qa(self, split):
        dataset = NarrativeQADataset(NarrativeQADatasetConfig(split=split))
        for item in dataset:
            assert isinstance(item, ContextualQASample)
            assert len(item.contexts) > 0
            assert isinstance(item.contexts[0], Context)
        print(f"NarrativeQA-{split} dataset length: {len(dataset)}")
        print(f"NarrativeQA-{split} dataset test passed.")
        return

    def test_simple_qa(self):
        dataset = SimpleQADataset(SimpleQADatasetConfig())
        for item in dataset:
            assert isinstance(item, QASample)
            assert hasattr(item, "meta_data")
        print(f"SimpleQA dataset length: {len(dataset)}")
        print("SimpleQA dataset test passed.")
        return

    def test_multihop_rag(self):
        dataset = MultihopRAGDataset(MultihopRAGDatasetConfig())
        for item in dataset:
            assert isinstance(item, ContextualQASample)
            assert hasattr(item, "meta_data")
        print(f"MultihopRAG dataset length: {len(dataset)}")
        print("MultihopRAG dataset test passed.")
        return

    @pytest.mark.parametrize("version", ["v1.1", "v2.0"])
    @pytest.mark.parametrize("split", ["train", "validation"])
    def test_squad(self, version, split):
        dataset = SQuADDataset(SQuADDatasetConfig(version=version, split=split))
        for item in dataset:
            assert isinstance(item, ContextualQASample)
            assert len(item.contexts) > 0
            assert isinstance(item.contexts[0], Context)
        print(f"SQuAD-{version}-{split} dataset length: {len(dataset)}")
        print(f"SQuAD-{version}-{split} dataset test passed.")
        return

    @pytest.mark.parametrize(
        "subset",
        [
            "questanswer_1doc",
            "questanswer_2docs",
            "questanswer_3docs",
        ],
    )
    def test_crud_qa(self, subset):
        dataset = CRUDQADataset(CRUDQADatasetConfig(subset=subset))
        for item in dataset:
            assert isinstance(item, ContextualQASample)
            assert len(item.contexts) > 0
            assert isinstance(item.contexts[0], Context)
        print(f"CRUD QA-{subset} dataset length: {len(dataset)}")
        print(f"CRUD QA-{subset} dataset test passed.")
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
            assert isinstance(item, IRSample)
            assert len(item.contexts + item.hard_negatives) > 0
        print(f"MLDR-{split}-{lang} dataset length: {len(dataset)}")
        print(f"MLDR-{split}-{lang} dataset test passed.")
        return


class TestMSMARCODataset:
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
                data_name="msmarco_document_ranking_v1",
                split=split,
                data_path=None,
                load_corpus=False,
            )
        )
        for item in dataset:
            pass
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
                data_name="msmarco_document_ranking_v2",
                split=split,
                data_path=None,
                load_corpus=False,
            )
        )
        for item in dataset:
            pass
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
                data_name="msmarco_passage_ranking_v1",
                split=split,
                data_path=None,
                load_corpus=False,
            )
        )
        for item in dataset:
            pass
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
                data_name="msmarco_passage_ranking_v2",
                split=split,
                data_path=None,
                load_corpus=False,
            )
        )
        for item in dataset:
            pass
        print(f"MSMARCO Passage Ranking V2 {split} split passed.")
        print(f"Number of samples: {len(dataset)}")
        return
