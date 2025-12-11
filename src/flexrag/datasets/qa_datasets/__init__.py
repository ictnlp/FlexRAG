from .attribute_qa import AttributedQADataset, AttributedQADatasetConfig
from .crud_qa import CRUDQADataset, CRUDQADatasetConfig
from .guten_qa import GutenQADataset, GutenQADatasetConfig
from .kilt_qa import KiltQADataset, KiltQADatasetConfig
from .literary_qa import LiteraryQADataset, LiteraryQADatasetConfig
from .multihop_rag import MultihopRAGDataset, MultihopRAGDatasetConfig
from .narrative_qa import NarrativeQADataset, NarrativeQADatasetConfig
from .qa_dataset_base import (
    KNOWLEDGE_QA_DATASETS,
    QA_DATASETS,
    KnowledgeQADatasetBase,
    KnowledgeQAEvalData,
    QADatasetBase,
    QAEvalData,
)
from .simple_qa import SimpleQADataset, SimpleQADatasetConfig
from .squad import SQuADDataset, SQuADDatasetConfig

QADatasetConfig = QA_DATASETS.make_config()


__all__ = [
    "AttributedQADataset",
    "AttributedQADatasetConfig",
    "CRUDQADataset",
    "CRUDQADatasetConfig",
    "GutenQADataset",
    "GutenQADatasetConfig",
    "KiltQADataset",
    "KiltQADatasetConfig",
    "LiteraryQADataset",
    "LiteraryQADatasetConfig",
    "MultihopRAGDataset",
    "MultihopRAGDatasetConfig",
    "NarrativeQADataset",
    "NarrativeQADatasetConfig",
    "KNOWLEDGE_QA_DATASETS",
    "QA_DATASETS",
    "KnowledgeQADatasetBase",
    "KnowledgeQAEvalData",
    "QADatasetBase",
    "QAEvalData",
    "SimpleQADataset",
    "SimpleQADatasetConfig",
    "SQuADDataset",
    "SQuADDatasetConfig",
    "QADatasetConfig",
]
