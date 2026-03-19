import orjson
import pytest

from flexrag.common import Context
from flexrag.datasets.benchmarks import (
    CRUDRAGDataset,
    CRUDRAGDatasetConfig,
    GutenQADataset,
    GutenQADatasetConfig,
    LongBenchDataset,
    LongBenchDatasetConfig,
    MemoryAgentBenchDataset,
    MemoryAgentBenchDatasetConfig,
    MultihopRAGDataset,
    MultihopRAGDatasetConfig,
    MuSiQueDataset,
    MuSiQueDatasetConfig,
    NarrativeQADataset,
    NarrativeQADatasetConfig,
    PerLTQADataset,
    PerLTQADatasetConfig,
    SQuADDataset,
    SQuADDatasetConfig,
)
from flexrag.datasets.core import ContextualQASample


class TestContextualQA:
    def valid_contextual_qa_sample(self, item, allow_empty_context: bool = False):
        assert isinstance(item, ContextualQASample)
        if not allow_empty_context:
            assert len(item.contexts) > 0
        if len(item.contexts) > 0:
            assert isinstance(item.contexts[0], Context)
        return

    @pytest.mark.parametrize("ctx_mode", ["lumber_chunk", "book"])
    def test_guten_qa(self, ctx_mode):
        dataset = GutenQADataset(GutenQADatasetConfig(context_mode=ctx_mode))
        for item in dataset:
            self.valid_contextual_qa_sample(item)
        print(f"GutenQA-{ctx_mode} dataset length: {len(dataset)}")
        print(f"GutenQA-{ctx_mode} dataset test passed.")
        return

    @pytest.mark.parametrize("split", ["train", "validation", "test"])
    def test_narrative_qa(self, split):
        dataset = NarrativeQADataset(NarrativeQADatasetConfig(split=split))
        for item in dataset:
            self.valid_contextual_qa_sample(item)
        print(f"NarrativeQA-{split} dataset length: {len(dataset)}")
        print(f"NarrativeQA-{split} dataset test passed.")
        return

    def test_multihop_rag(self):
        dataset = MultihopRAGDataset(MultihopRAGDatasetConfig())
        for item in dataset:
            self.valid_contextual_qa_sample(item, True)
        print(f"MultihopRAG dataset length: {len(dataset)}")
        print("MultihopRAG dataset test passed.")
        return

    @pytest.mark.parametrize(
        ("split", "record"),
        [
            (
                "validation",
                {
                    "id": "2hop__10_20",
                    "question": "Which city is the birthplace of the author of Example Book?",
                    "answer": "Example City",
                    "answer_aliases": ["Example City", "City of Example"],
                    "answerable": True,
                    "question_decomposition": [
                        {
                            "id": "10",
                            "question": "Who wrote Example Book?",
                            "answer": "Alice Writer",
                            "paragraph_support_idx": 0,
                        },
                        {
                            "id": "20",
                            "question": "Where was Alice Writer born?",
                            "answer": "Example City",
                            "paragraph_support_idx": 1,
                        },
                    ],
                    "paragraphs": [
                        {
                            "idx": 0,
                            "title": "Example Book",
                            "paragraph_text": "Example Book was written by Alice Writer.",
                            "is_supporting": True,
                        },
                        {
                            "idx": 1,
                            "title": "Alice Writer",
                            "paragraph_text": "Alice Writer was born in Example City.",
                            "is_supporting": True,
                        },
                        {
                            "idx": 2,
                            "title": "Distractor",
                            "paragraph_text": "This paragraph is unrelated.",
                            "is_supporting": False,
                        },
                    ],
                },
            ),
            (
                "test",
                {
                    "id": "2hop__30_40",
                    "question": "Which river flows through Example City?",
                    "paragraphs": [
                        {
                            "idx": 0,
                            "title": "Example City",
                            "paragraph_text": "Example City is crossed by the River Sample.",
                        }
                    ],
                },
            ),
        ],
    )
    def test_musique(self, tmp_path, split, record):
        file_name_map = {
            "validation": "musique_full_v1.0_dev.jsonl",
            "test": "musique_full_v1.0_test.jsonl",
        }
        data_path = tmp_path / file_name_map[split]
        data_path.write_bytes(orjson.dumps(record) + b"\n")

        dataset = MuSiQueDataset(
            MuSiQueDatasetConfig(data_path=tmp_path.as_posix(), split=split)
        )
        assert len(dataset) == 1
        item = dataset[0]
        self.valid_contextual_qa_sample(item)
        assert item.question_id == record["id"]
        assert item.question == record["question"]
        assert item.meta_data["question_decomposition"] == record.get(
            "question_decomposition", []
        )
        assert item.meta_data["answerable"] == record.get("answerable")

        support_indices = [
            context.meta_data["idx"]
            for context in item.contexts
            if context.meta_data["is_supporting"]
        ]
        assert item.meta_data["supporting_paragraph_indices"] == support_indices

        expected_answers = []
        if "answer" in record:
            expected_answers.append(record["answer"])
        for alias in record.get("answer_aliases", []):
            if alias not in expected_answers:
                expected_answers.append(alias)
        assert item.answers == expected_answers
        print(f"MuSiQue-{split} dataset length: {len(dataset)}")
        print(f"MuSiQue-{split} dataset test passed.")
        return

    @pytest.mark.parametrize("version", ["v1.1", "v2.0"])
    @pytest.mark.parametrize("split", ["train", "validation"])
    def test_squad(self, version, split):
        dataset = SQuADDataset(SQuADDatasetConfig(version=version, split=split))
        for item in dataset:
            self.valid_contextual_qa_sample(item)
        print(f"SQuAD-{version}-{split} dataset length: {len(dataset)}")
        print(f"SQuAD-{version}-{split} dataset test passed.")
        return

    @pytest.mark.parametrize(
        "subset",
        [
            "questanswer_1doc",
            "questanswer_2docs",
            "questanswer_3docs",
        ],
    )
    def test_crud_qa(self, subset):
        dataset = CRUDRAGDataset(CRUDRAGDatasetConfig(subset=subset))
        for item in dataset:
            self.valid_contextual_qa_sample(item)
        print(f"CRUD QA-{subset} dataset length: {len(dataset)}")
        print(f"CRUD QA-{subset} dataset test passed.")
        return

    @pytest.mark.parametrize(
        "subset",
        [
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
        ],
    )
    def test_longbench(self, subset):
        dataset = LongBenchDataset(LongBenchDatasetConfig(subset=subset))
        for item in dataset:
            self.valid_contextual_qa_sample(item)
        print(f"LongBench-{subset} dataset length: {len(dataset)}")
        print(f"LongBench-{subset} dataset test passed.")
        return

    @pytest.mark.parametrize(
        "split",
        [
            "Accurate_Retrieval",
            "Test_Time_Learning",
            "Long_Range_Understanding",
            "Conflict_Resolution",
        ],
    )
    def test_memory_agent_bench(self, split):
        dataset = MemoryAgentBenchDataset(MemoryAgentBenchDatasetConfig(split=split))
        for item in dataset:
            self.valid_contextual_qa_sample(item)
        print(f"MemoryAgentBench-{split} dataset length: {len(dataset)}")
        print(f"MemoryAgentBench-{split} dataset test passed.")
        return

    @pytest.mark.parametrize(
        "lang",
        ["en", "en_v2", "zh"],
    )
    def test_perltqa(self, lang):
        dataset = PerLTQADataset(PerLTQADatasetConfig(lang=lang))
        for item in dataset:
            self.valid_contextual_qa_sample(item)
        print(f"PerLTQA-{lang} dataset length: {len(dataset)}")
        print(f"PerLTQA-{lang} dataset test passed.")
        return
