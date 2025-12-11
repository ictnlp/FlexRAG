from .long_bench import LongBenchMCDataset, LongBenchMCDatasetConfig
from .long_bench_v2 import LongBenchV2Dataset, LongBenchV2DatasetConfig
from .multiple_choice_dataset_base import (
    KNOWLEDGE_MULTIPLE_CHOICE_DATASETS,
    MULTIPLE_CHOICE_DATASETS,
    KnowledgeMultipleChoiceData,
    KnowledgeMultipleChoiceDatasetBase,
    MultipleChoiceDatasetBase,
    MultipleChoiceEvalData,
)
from .novel_qa import NovelQAConfig, NovelQADataset
from .quality import QuALITYDataset, QuALITYDatasetConfig

KnowledgeMultipleChoiceDatasetConfig = KNOWLEDGE_MULTIPLE_CHOICE_DATASETS.make_config()
MultipleChoiceDatasetConfig = MULTIPLE_CHOICE_DATASETS.make_config()


__all__ = [
    "LongBenchMCDataset",
    "LongBenchMCDatasetConfig",
    "LongBenchV2Dataset",
    "LongBenchV2DatasetConfig",
    "KNOWLEDGE_MULTIPLE_CHOICE_DATASETS",
    "MULTIPLE_CHOICE_DATASETS",
    "KnowledgeMultipleChoiceData",
    "KnowledgeMultipleChoiceDatasetBase",
    "MultipleChoiceDatasetBase",
    "MultipleChoiceEvalData",
    "NovelQAConfig",
    "NovelQADataset",
    "QuALITYDataset",
    "QuALITYDatasetConfig",
    "KnowledgeMultipleChoiceDatasetConfig",
    "MultipleChoiceDatasetConfig",
]
