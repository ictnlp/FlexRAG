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
from flexrag.tasks.deep_search.gisa import _GISAOfficialMetric

pytestmark = pytest.mark.integration


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
            assert "sample_id" in item.metadata
            assert "split" in item.metadata
            assert "qa_pairs" in item.metadata
            assert "wikipages" in item.metadata
            assert "annotations" in item.metadata
            assert isinstance(item.metadata["qa_pairs"], list)
            assert isinstance(item.metadata["wikipages"], list)
            assert isinstance(item.metadata["annotations"], list)
        print(f"ASQA dataset length: {len(dataset)}")
        print("ASQA dataset test passed.")
        return

    def test_popqa(self):
        dataset = PopQADataset(PopQADatasetConfig())
        for item in dataset:
            self.valid_qa_sample(item)
            assert item.question_id is not None
            assert len(item.answers) > 0
            assert "subj" in item.metadata
            assert "prop" in item.metadata
            assert "obj" in item.metadata
            assert "possible_answers" in item.metadata
            assert isinstance(item.metadata["possible_answers"], list)
            assert isinstance(item.metadata["s_aliases"], list)
            assert isinstance(item.metadata["o_aliases"], list)
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
        samples_by_type = {}
        for item in dataset:
            self.valid_qa_sample(item)
            assert item.question_id is not None
            assert len(item.answers) > 0
            assert item.answers[0].strip()
            assert Path(item.metadata["answer_path"]).exists()
            samples_by_type.setdefault(item.metadata["answer_type"], item)
        assert {"item", "set", "list", "table"}.issubset(samples_by_type)
        for answer_type, item in samples_by_type.items():
            assert not item.answers[0].lstrip().startswith("[")
            if answer_type != "table":
                assert item.answers[0].splitlines()[0].strip()
        print(f"GISA dataset length: {len(dataset)}")
        print("GISA dataset test passed.")
        return

    def test_gisa_official_metric(self):
        metric = _GISAOfficialMetric()
        scores, details = metric(
            responses=[
                "<answer>\n```tsv\nValue\nKansas City Chiefs\n```\n</answer>",
                "```tsv\nItem\nA\nB\n```",
                "```tsv\nItem\nA\nC\n```",
                "```tsv\nName\tRole\nAlice\tEngineer\nBob\tDesigner\n```",
            ],
            golden_responses=[
                ["Kansas City Chiefs\n"],
                ["A\nB\n"],
                ["A\nB\nC\n"],
                ["Name,Role\nAlice,Engineer\nBob,Designer\n"],
            ],
            metadatas=[
                {"id": "item", "answer_type": "item"},
                {"id": "set", "answer_type": "set"},
                {"id": "list", "answer_type": "list"},
                {"id": "table", "answer_type": "table"},
            ],
        )
        assert scores["overall_global_em"] == 0.75
        assert details["summary"]["item"]["overall_item_em"] == 1.0
        assert details["summary"]["set"]["overall_set_f1"] == 1.0
        assert details["summary"]["list"]["overall_list_content_f1"] == 0.8
        assert details["summary"]["table"]["overall_table_row_f1"] == 1.0
        assert details["summary"]["table"]["overall_table_item_f1"] == 1.0
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
            assert item.metadata["subset"] == subset
            assert "doc_name" in item.metadata
            assert Path(item.metadata["source_file_path"]).exists()
            assert item.metadata["source_file_format"] == "pdf"
        print(f"UDA-QA-{subset} dataset length: {len(dataset)}")
        print(f"UDA-QA-{subset} dataset test passed.")
        return
