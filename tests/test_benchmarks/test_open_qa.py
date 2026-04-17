from pathlib import Path

import pytest

from flexrag.datasets.benchmarks import (
    ASQADataset,
    ASQADatasetConfig,
    BrowseCompDataset,
    BrowseCompDatasetConfig,
    BrowseCompZHDataset,
    BrowseCompZHDatasetConfig,
    DeepResearch9KDataset,
    DeepResearch9KDatasetConfig,
    DeepSearchQADataset,
    DeepSearchQADatasetConfig,
    GAIADataset,
    GAIADatasetConfig,
    GISADataset,
    GISADatasetConfig,
    MedBrowseCompDataset,
    MedBrowseCompDatasetConfig,
    PopQADataset,
    PopQADatasetConfig,
    SimpleQADataset,
    SimpleQADatasetConfig,
    UDAQADataset,
    UDAQADatasetConfig,
    WideSearchDataset,
    WideSearchDatasetConfig,
)
from flexrag.datasets.core import QASample


class TestOpenDomainQA:
    def valid_qa_sample(self, item):
        assert isinstance(item, QASample)
        return

    def test_deepsearch_qa(self):
        dataset = DeepSearchQADataset(DeepSearchQADatasetConfig())
        for item in dataset:
            self.valid_qa_sample(item)
        print(f"DeepSearch QA dataset length: {len(dataset)}")
        print("DeepSearch QA dataset test passed.")
        return

    def test_browsecomp(self):
        dataset = BrowseCompDataset(BrowseCompDatasetConfig())
        for item in dataset:
            self.valid_qa_sample(item)
        print(f"BrowseComp dataset length: {len(dataset)}")
        print("BrowseComp dataset test passed.")
        return

    def test_browsecomp_zh(self):
        dataset = BrowseCompZHDataset(BrowseCompZHDatasetConfig())
        for item in dataset:
            self.valid_qa_sample(item)
        print(f"BrowseComp-ZH dataset length: {len(dataset)}")
        print("BrowseComp-ZH dataset test passed.")
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
            self.valid_qa_sample(item)
        print(f"GAIA-{subset}-{split} dataset length: {len(dataset)}")
        print(f"GAIA-{subset}-{split} dataset test passed.")
        return

    def test_simple_qa(self):
        dataset = SimpleQADataset(SimpleQADatasetConfig())
        for item in dataset:
            self.valid_qa_sample(item)
        print(f"SimpleQA dataset length: {len(dataset)}")
        print("SimpleQA dataset test passed.")
        return

    def test_asqa(self):
        dataset = ASQADataset(ASQADatasetConfig())
        for item in dataset:
            self.valid_qa_sample(item)
            assert item.question_id is not None
            assert len(item.answers) > 0
            assert "sample_id" in item.meta_data
            assert "split" in item.meta_data
            assert "qa_pairs" in item.meta_data
            assert "wikipages" in item.meta_data
            assert "annotations" in item.meta_data
            assert isinstance(item.meta_data["qa_pairs"], list)
            assert isinstance(item.meta_data["wikipages"], list)
            assert isinstance(item.meta_data["annotations"], list)
        print(f"ASQA dataset length: {len(dataset)}")
        print("ASQA dataset test passed.")
        return

    def test_popqa(self):
        dataset = PopQADataset(PopQADatasetConfig())
        for item in dataset:
            self.valid_qa_sample(item)
            assert item.question_id is not None
            assert len(item.answers) > 0
            assert "subj" in item.meta_data
            assert "prop" in item.meta_data
            assert "obj" in item.meta_data
            assert "possible_answers" in item.meta_data
            assert isinstance(item.meta_data["possible_answers"], list)
            assert isinstance(item.meta_data["s_aliases"], list)
            assert isinstance(item.meta_data["o_aliases"], list)
        print(f"PopQA dataset length: {len(dataset)}")
        print("PopQA dataset test passed.")
        return

    @pytest.mark.parametrize(
        "subset",
        ["50", "605", "cua"],
    )
    def test_med_browsecomp(self, subset):
        dataset = MedBrowseCompDataset(MedBrowseCompDatasetConfig(subset=subset))
        for item in dataset:
            self.valid_qa_sample(item)
        print(f"MedBrowseComp-{subset} dataset length: {len(dataset)}")
        print(f"MedBrowseComp-{subset} dataset test passed.")
        return

    def test_deepresearch_9k(self):
        dataset = DeepResearch9KDataset(DeepResearch9KDatasetConfig())
        for item in dataset:
            self.valid_qa_sample(item)
        print(f"DeepResearch9K dataset length: {len(dataset)}")
        print("DeepResearch9K dataset test passed.")
        return

    def test_wide_search(self):
        dataset = WideSearchDataset(WideSearchDatasetConfig())
        for item in dataset:
            self.valid_qa_sample(item)
        print(f"WideSearch dataset length: {len(dataset)}")
        print("WideSearch dataset test passed.")
        return

    def test_gisa(self):
        dataset = GISADataset(GISADatasetConfig())
        for item in dataset:
            self.valid_qa_sample(item)
        print(f"GISA dataset length: {len(dataset)}")
        print("GISA dataset test passed.")
        return

    @pytest.mark.parametrize(
        "subset",
        ["feta", "nq", "paper_text", "paper_tab", "fin", "tat"],
    )
    def test_uda_qa(self, subset):
        dataset = UDAQADataset(UDAQADatasetConfig(subset=subset))
        for item in dataset:
            self.valid_qa_sample(item)
            assert item.question_id is not None
            assert len(item.answers) > 0
            assert item.meta_data["subset"] == subset
            assert "doc_name" in item.meta_data
            assert Path(item.meta_data["source_file_path"]).exists()
            assert item.meta_data["source_file_format"] == "pdf"
        print(f"UDA-QA-{subset} dataset length: {len(dataset)}")
        print(f"UDA-QA-{subset} dataset test passed.")
        return
