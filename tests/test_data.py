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
    MultihopRAGDataset,
    MultihopRAGDatasetConfig,
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
    MultiLongDocRetrievalDataset,
    MultiLongDocRetrievalDatasetConfig,
)
from flexrag.datasets.core import (
    ContextualQASample,
    QASample,
    ContextualMCSample,
    IRSample,
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

    def test_gaia(self):
        for subset in ["2023_all", "2023_level1", "2023_level2", "2023_level3"]:
            for split in ["validation", "test"]:
                dataset = GAIADataset(GAIADatasetConfig())
                for item in dataset:
                    assert isinstance(item, QASample)
                print(f"GAIA-{subset}-{split} dataset length: {len(dataset)}")
                print(f"GAIA-{subset}-{split} dataset test passed.")
        return

    def test_longbench(self):
        subsets = [
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
        ]
        for subset in subsets:
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

    def test_quality(self):
        for split in ["train", "validation", "test"]:
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

    def test_kilt(self):
        subsets = [
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
        ]
        for subset in subsets:
            for split in ["validation", "test"]:
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

    def test_guten_qa(self):
        for ctx_mode in ["lumber_chunk", "book"]:
            dataset = GutenQADataset(GutenQADatasetConfig(context_mode=ctx_mode))
            for item in dataset:
                assert isinstance(item, ContextualQASample)
                assert len(item.contexts) > 0
                assert isinstance(item.contexts[0], Context)
            print(f"GutenQA-{ctx_mode} dataset length: {len(dataset)}")
            print(f"GutenQA-{ctx_mode} dataset test passed.")
        return

    def test_narrative_qa(self):
        for split in ["train", "validation", "test"]:
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

    def test_squad(self):
        for version in ["v1.1", "v2.0"]:
            for split in ["train", "validation"]:
                dataset = SQuADDataset(SQuADDatasetConfig(version=version, split=split))
                for item in dataset:
                    assert isinstance(item, ContextualQASample)
                    assert len(item.contexts) > 0
                    assert isinstance(item.contexts[0], Context)
                print(f"SQuAD-{version}-{split} dataset length: {len(dataset)}")
                print(f"SQuAD-{version}-{split} dataset test passed.")
        return

    def test_crud_qa(self):
        for subset in ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]:
            dataset = CRUDQADataset(CRUDQADatasetConfig(subset=subset))
            for item in dataset:
                assert isinstance(item, ContextualQASample)
                assert len(item.contexts) > 0
                assert isinstance(item.contexts[0], Context)
            print(f"CRUD QA-{subset} dataset length: {len(dataset)}")
            print(f"CRUD QA-{subset} dataset test passed.")
        return

    def test_mldr(self):
        for split in ["train", "dev", "test"]:
            for lang in [
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
            ]:
                dataset = MultiLongDocRetrievalDataset(
                    MultiLongDocRetrievalDatasetConfig(split=split, lang=lang)
                )
                for item in dataset:
                    assert isinstance(item, IRSample)
                    assert len(item.contexts + item.hard_negatives) > 0
                print(f"MLDR-{split}-{lang} dataset length: {len(dataset)}")
                print(f"MLDR-{split}-{lang} dataset test passed.")
        return
