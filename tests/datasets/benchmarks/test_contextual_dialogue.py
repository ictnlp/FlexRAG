import pytest

from flexrag.common import Context
from flexrag.datasets.benchmarks import (
    ChatRAGBenchDataset,
    ChatRAGBenchDatasetConfig,
)
from flexrag.datasets.core import ContextualDialogueSample

pytestmark = pytest.mark.integration


class TestContextualDialogue:
    def valid_contextual_dialogue_sample(self, item):
        assert isinstance(item, ContextualDialogueSample)
        assert item.dialogue_id is not None
        assert item.messages is not None
        assert len(item.golden_responses) > 0
        assert len(item.contexts) > 0
        assert isinstance(item.contexts[0], Context)
        return

    @pytest.mark.parametrize("subset", ["coqa", "all"])
    def test_chatrag_bench(self, subset):
        dataset = ChatRAGBenchDataset(
            ChatRAGBenchDatasetConfig(subset=subset, num_ctx=2)
        )
        for item in dataset:
            self.valid_contextual_dialogue_sample(item)
        print(f"ChatRAG-Bench-{subset} dataset length: {len(dataset)}")
        print(f"ChatRAG-Bench-{subset} dataset test passed.")
        return
