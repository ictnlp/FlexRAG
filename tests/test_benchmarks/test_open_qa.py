import pytest

from flexrag.datasets.benchmarks import (
    BrowseCompDataset,
    BrowseCompDatasetConfig,
    DeepSearchQADataset,
    DeepSearchQADatasetConfig,
    GAIADataset,
    GAIADatasetConfig,
    PopQADataset,
    PopQADatasetConfig,
    SimpleQADataset,
    SimpleQADatasetConfig,
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
