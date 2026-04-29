import pytest

from flexrag.datasets.benchmarks import (
    BrowseCompPlusDataset,
    BrowseCompPlusDatasetConfig,
    FramesDataset,
    FramesDatasetConfig,
    MTRAGDataset,
    MTRAGDatasetConfig,
)
from flexrag.datasets.core import IRDialogueSample, IRQASample


class TestEndToEndRAG:
    def valid_ir_qa_sample(self, item):
        assert isinstance(item, IRQASample)
        assert item.question_id is not None
        return

    def valid_ir_dialogue_sample(self, item):
        assert isinstance(item, IRDialogueSample)
        assert item.dialogue_id is not None
        assert item.messages is not None
        return

    @pytest.mark.parametrize(
        "subset",
        ["clapnq", "cloud", "fiqa", "govt", "all"],
    )
    def test_mtrag(self, subset):
        dataset = MTRAGDataset(MTRAGDatasetConfig(subset=subset))
        for item in dataset:
            self.valid_ir_dialogue_sample(item)
        print(f"MTRAG-{subset} dataset length: {len(dataset)}")
        print(f"MTRAG-{subset} dataset test passed.")
        return

    def test_frames(self):
        dataset = FramesDataset(FramesDatasetConfig(load_corpus=False))
        for item in dataset:
            self.valid_ir_qa_sample(item)
        print(f"Frames dataset length: {len(dataset)}")
        print("Frames dataset test passed.")
        return

    def test_browsecomp_plus(self):
        dataset = BrowseCompPlusDataset(BrowseCompPlusDatasetConfig(load_corpus=False))
        for item in dataset:
            self.valid_ir_qa_sample(item)
        print(f"BrowseCompPlus dataset length: {len(dataset)}")
        print("BrowseCompPlus dataset test passed.")
        return
