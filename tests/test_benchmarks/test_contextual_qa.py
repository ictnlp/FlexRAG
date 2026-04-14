import json

import pytest

from flexrag.common import Context
from flexrag.datasets.benchmarks import (
    CRUDRAGDataset,
    CRUDRAGDatasetConfig,
    GutenQADataset,
    GutenQADatasetConfig,
    LongBenchDataset,
    LongBenchDatasetConfig,
    LoongDataset,
    LoongDatasetConfig,
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
    QasperDataset,
    QasperDatasetConfig,
    SQuADDataset,
    SQuADDatasetConfig,
    TwoWikiMultihopQADataset,
    TwoWikiMultihopQADatasetConfig,
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

    def test_loong(self):
        dataset = LoongDataset(LoongDatasetConfig())
        for item in dataset:
            self.valid_contextual_qa_sample(item)
        print(f"Loong dataset length: {len(dataset)}")
        print("Loong dataset test passed.")
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

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_twowiki_multihop_qa(self, split):
        dataset = TwoWikiMultihopQADataset(TwoWikiMultihopQADatasetConfig(split=split))
        for item in dataset:
            self.valid_contextual_qa_sample(item)
            assert item.question_id is not None
            assert item.meta_data is not None
            assert "supporting_facts" in item.meta_data
            assert "evidences" in item.meta_data
        print(f"2WikiMultihopQA-{split} dataset length: {len(dataset)}")
        print(f"2WikiMultihopQA-{split} dataset test passed.")
        return

    @pytest.mark.parametrize("split", ["train", "validation", "test"])
    def test_musique(self, split):
        dataset = MuSiQueDataset(MuSiQueDatasetConfig(split=split))
        for item in dataset:
            self.valid_contextual_qa_sample(item)
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

    @pytest.mark.parametrize("split", ["train", "validation", "test"])
    @pytest.mark.parametrize("context_mode", ["paragraph", "paper"])
    def test_qasper(self, tmp_path, split, context_mode):
        data = {
            "paper-1": {
                "title": "Paper Title",
                "abstract": "Paper Abstract",
                "full_text": [
                    {
                        "section_name": "Introduction",
                        "paragraphs": ["Paragraph 1.", "   ", "Paragraph 2."],
                    }
                ],
                "figures_and_tables": [
                    {"caption": "Figure caption.", "file": "fig1.png"},
                    {"caption": "   ", "file": "fig2.png"},
                ],
                "qas": [
                    {
                        "question_id": "q1",
                        "question": "What is the answer?",
                        "question_writer": "author",
                        "paper_read": "full",
                        "search_query": "query",
                        "topic_background": "high",
                        "nlp_background": "high",
                        "answers": [
                            {
                                "annotation_id": "a1",
                                "worker_id": "w1",
                                "answer": {
                                    "unanswerable": False,
                                    "yes_no": None,
                                    "free_form_answer": "Free form answer",
                                    "extractive_spans": ["Span 1", "Span 1"],
                                    "evidence": [],
                                    "highlighted_evidence": [],
                                },
                            }
                        ],
                    },
                    {
                        "question_id": "q2",
                        "question": "Is it true?",
                        "question_writer": "author",
                        "paper_read": "full",
                        "search_query": "query",
                        "topic_background": "high",
                        "nlp_background": "high",
                        "answers": [
                            {
                                "annotation_id": "a2",
                                "worker_id": "w2",
                                "answer": {
                                    "unanswerable": False,
                                    "yes_no": False,
                                    "free_form_answer": "",
                                    "extractive_spans": [],
                                    "evidence": [],
                                    "highlighted_evidence": [],
                                },
                            }
                        ],
                    },
                    {
                        "question_id": "q3",
                        "question": "What is missing?",
                        "question_writer": "author",
                        "paper_read": "full",
                        "search_query": "query",
                        "topic_background": "high",
                        "nlp_background": "high",
                        "answers": [
                            {
                                "annotation_id": "a3",
                                "worker_id": "w3",
                                "answer": {
                                    "unanswerable": True,
                                    "yes_no": None,
                                    "free_form_answer": "",
                                    "extractive_spans": [],
                                    "evidence": [],
                                    "highlighted_evidence": [],
                                },
                            }
                        ],
                    },
                ],
            }
        }
        for file_name in [
            "qasper-train-v0.3.json",
            "qasper-dev-v0.3.json",
            "qasper-test-v0.3.json",
        ]:
            (tmp_path / file_name).write_text(
                json.dumps(data, ensure_ascii=False),
                encoding="utf-8",
            )

        dataset = QasperDataset(
            QasperDatasetConfig(
                data_path=tmp_path.as_posix(),
                split=split,
                context_mode=context_mode,
            )
        )
        assert len(dataset) == 3

        item = dataset[0]
        self.valid_contextual_qa_sample(item)
        assert item.meta_data is not None
        assert item.meta_data["title"] == "Paper Title"
        assert item.meta_data["abstract"] == "Paper Abstract"
        if context_mode == "paragraph":
            assert len(item.contexts) == 3
            assert item.contexts[-1].meta_data["kind"] == "figure_or_table"
        else:
            assert len(item.contexts) == 1
            assert (
                item.contexts[0].data["text"]
                == "Paragraph 1.\nParagraph 2.\nFigure caption."
            )
        assert dataset[1].answers == ["no"]
        assert dataset[2].answers == ["unanswerable"]
        print(f"Qasper-{split}-{context_mode} dataset length: {len(dataset)}")
        print(f"Qasper-{split}-{context_mode} dataset test passed.")
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
