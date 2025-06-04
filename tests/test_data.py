import pytest

from flexrag.datasets import (
    RAGEvalData,
    RAGEvalDataset,
    RAGEvalDatasetConfig,
    RAGMultipleChoiceData,
)
from flexrag.utils import LOGGER_MANAGER

logger = LOGGER_MANAGER.get_logger("tests.datasets")


class TestRAGEvalDataset:
    datasets = {
        # for simplicity, we only test one dataset.
        "2wikimultihopqa": ["dev", "train"],
        # "ambig_qa": ["dev", "train"],
        # "arc": ["dev", "test", "train"],
        # "asqa": ["dev", "train"],
        # "ay2": ["dev", "train"],
        # "bamboogle": ["test"],
        # "boolq": ["dev", "train"],  # True/False task is not supported yet
        # "commonsenseqa": ["dev", "train"],
        # "curatedtrec": ["test", "train"],
        # "domainrag": ["test"],  # Error in loading due to dataset schema
        # "eli5": ["dev", "train"],
        # "fermi": ["dev", "test", "train"],
        # "fever": ["dev", "train"],
        # "hellaswag": ["dev", "train"],
        # "hotpotqa": ["dev", "train"],
        # "mmlu": ["5_shot", "dev", "test", "train"],
        # "msmarco-qa": ["dev", "train"],
        # "musique": ["dev", "train"],
        # "narrativeqa": ["dev", "test", "train"],
        # "nq": ["dev", "test", "train"],
        # "openbookqa": ["dev", "test", "train"],
        # "piqa": ["dev", "train"],
        # "popqa": ["test"],
        # "quartz": ["dev", "test", "train"],
        # "siqa": ["dev", "train"],
        # "squad": ["dev", "train"],
        # "t-rex": ["dev", "train"],
        # "triviaqa": ["dev", "test", "train"],
        # "truthful_qa": ["dev"],
        # "web_questions": ["test", "train"],
        # "wikiasp": ["dev", "test", "train"],
        # "wikiqa": ["dev", "test", "train"],
        # "wned": ["dev"],
        # "wow": ["dev", "train"],
        # "zero-shot_re": ["dev", "train"],
    }

    async def run_test(self, name: str, split: str):
        # load dataset
        dataset = RAGEvalDataset(RAGEvalDatasetConfig(name=name, split=split))

        # check dataset
        assert len(dataset) > 0
        for i in dataset:
            assert isinstance(i, (RAGEvalData, RAGMultipleChoiceData))
        for i in range(len(dataset)):
            assert isinstance(dataset[i], (RAGEvalData, RAGMultipleChoiceData))
        return

    @pytest.mark.asyncio
    async def test_rageval_dataset(self):
        logger.info("Testing RAGEvalDataset")
        for name in self.datasets:
            for split in self.datasets[name]:
                logger.info(f"Testing {name} {split}")
                await self.run_test(name, split)
        return
