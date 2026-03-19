import pytest

from flexrag.datasets.benchmarks import (
    BrowseCompDataset,
    BrowseCompDatasetConfig,
    DeepResearch9KDataset,
    DeepResearch9KDatasetConfig,
    DeepSearchQADataset,
    DeepSearchQADatasetConfig,
    GAIADataset,
    GAIADatasetConfig,
    MedBrowseCompDataset,
    MedBrowseCompDatasetConfig,
    PopQADataset,
    PopQADatasetConfig,
    SimpleQADataset,
    SimpleQADatasetConfig,
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
