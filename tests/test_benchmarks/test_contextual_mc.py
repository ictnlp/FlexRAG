import pytest

from flexrag.common import Context
from flexrag.datasets.benchmarks import (
    LongBenchV2Dataset,
    LongBenchV2DatasetConfig,
    NovelQADataset,
    NovelQADatasetConfig,
    QuALITYDataset,
    QuALITYDatasetConfig,
)
from flexrag.datasets.core import ContextualMCSample


class TestContextualMC:
    def valid_contextual_mc_sample(self, item):
        assert isinstance(item, ContextualMCSample)
        assert len(item.contexts) > 0
        assert isinstance(item.contexts[0], Context)
        return

    @pytest.mark.parametrize("split", ["train", "validation", "test"])
    def test_quality(self, split):
        dataset = QuALITYDataset(QuALITYDatasetConfig(split=split))
        for item in dataset:
            self.valid_contextual_mc_sample(item)
        print(f"QuALITY-{split} dataset length: {len(dataset)}")
        print(f"QuALITY-{split} dataset test passed.")
        return

    def test_novel_qa(self):
        dataset = NovelQADataset(NovelQADatasetConfig())
        for item in dataset:
            self.valid_contextual_mc_sample(item)
        print(f"NovelQA dataset length: {len(dataset)}")
        print("NovelQA dataset test passed.")
        return

    def test_longbench_v2(self):
        dataset = LongBenchV2Dataset(LongBenchV2DatasetConfig())
        for item in dataset:
            self.valid_contextual_mc_sample(item)
        print(f"LongBenchV2 dataset length: {len(dataset)}")
        print("LongBenchV2 dataset test passed.")
        return
