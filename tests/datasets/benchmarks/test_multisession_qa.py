import pytest

from flexrag.datasets.benchmarks import (
    ConvoMemDataset,
    ConvoMemDatasetConfig,
    LoCoMoDataset,
    LoCoMoDatasetConfig,
    LongMemEvalDataset,
    LongMemEvalDatasetConfig,
    MSCSelfInstructDataset,
    MSCSelfInstructDatasetConfig,
)
from flexrag.datasets.core import MultiSessionQASample

pytestmark = pytest.mark.integration


class TestMultiSessionQADatasets:
    def valid_multisession_qa_sample(self, sample):
        assert isinstance(sample, MultiSessionQASample)
        assert all(isinstance(session.metadata, dict) for session in sample.sessions)
        return

    @pytest.mark.parametrize(
        "subset",
        [
            "abstention_evidence",  # aligned
            "assistant_facts_evidence",  # 13,797 vs 12,745
            "changing_evidence",  # aligned
            "implicit_connection_evidence",  # 7,746 vs 7,546
            "preference_evidence",  # 5,979 vs 5,079
            "user_evidence",  # aligned
        ],
    )
    def test_convo_mem(self, subset):
        dataset = ConvoMemDataset(ConvoMemDatasetConfig(subset=subset))
        for item in dataset:
            self.valid_multisession_qa_sample(item)
            assert all(
                {"session_id", "generator"} <= session.metadata.keys()
                for session in item.sessions
            )
            assert "session_annotations" in item.metadata
        print(f"ConvoMem-{subset} dataset length: {len(dataset)}")
        print(f"ConvoMem-{subset} dataset test passed.")
        return

    def test_msc_self_instruct(self):
        dataset = MSCSelfInstructDataset(MSCSelfInstructDatasetConfig())
        for item in dataset:
            self.valid_multisession_qa_sample(item)
        print(f"MSC-Self-Instruct dataset length: {len(dataset)}")
        print("MSC-Self-Instruct dataset test passed.")
        return

    @pytest.mark.parametrize(
        "split",
        ["oracle", "s_cleaned", "m_cleaned"],
    )
    def test_long_mem_eval(self, split):
        dataset = LongMemEvalDataset(LongMemEvalDatasetConfig(split=split))
        for item in dataset:
            self.valid_multisession_qa_sample(item)
            assert item.sessions_id == item.question_id
            assert all(
                {"session_id", "date"} <= session.metadata.keys()
                for session in item.sessions
            )
        print(f"LongMemEval-{split} dataset length: {len(dataset)}")
        print(f"LongMemEval-{split} dataset test passed.")
        return

    def test_locomo(self):
        dataset = LoCoMoDataset(LoCoMoDatasetConfig())
        for item in dataset:
            self.valid_multisession_qa_sample(item)
            assert all("session_id" in session.metadata for session in item.sessions)
            assert "session_annotations" in item.metadata
        print(f"LoCoMo dataset length: {len(dataset)}")
        print("LoCoMo dataset test passed.")
        return
