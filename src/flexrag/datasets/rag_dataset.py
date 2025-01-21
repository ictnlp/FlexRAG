from dataclasses import dataclass, field
from typing import Iterator, Optional

from flexrag.common_dataclass import Context, RAGEvalData
from flexrag.text_process import TextProcessPipeline, TextProcessPipelineConfig
from flexrag.utils import LOGGER_MANAGER

from .line_delimited_dataset import LineDelimitedDataset, LineDelimitedDatasetConfig

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.rag_dataset")


RAGEvalDatasetConfig = LineDelimitedDatasetConfig


class RAGEvalDataset(LineDelimitedDataset):
    def __iter__(self) -> Iterator[RAGEvalData]:
        for data in super().__iter__():
            golden_contexts = data.pop("golden_contexts", None)
            golden_contexts = (
                [Context(**context) for context in golden_contexts]
                if golden_contexts is not None
                else None
            )
            formatted_data = RAGEvalData(
                question=data.pop("question"),
                golden_contexts=golden_contexts,
                golden_answers=data.pop("golden_answers", None),
            )
            formatted_data.meta_data = data.pop("meta_data", {})
            formatted_data.meta_data.update(data)
            yield formatted_data


@dataclass
class RAGCorpusDatasetConfig(LineDelimitedDatasetConfig):
    saving_fields: list[str] = field(default_factory=list)
    id_field: Optional[str] = None
    text_process_pipeline: TextProcessPipelineConfig = field(default_factory=TextProcessPipelineConfig)  # type: ignore
    text_process_fields: list[str] = field(default_factory=list)


class RAGCorpusDataset(LineDelimitedDataset):
    def __init__(self, cfg: RAGCorpusDatasetConfig) -> None:
        super().__init__(cfg)
        # load arguments
        self.saving_fields = cfg.saving_fields
        self.id_field = cfg.id_field
        if self.id_field is None:
            logger.warning("No id field is provided, using the index as the id field")

        # load text pre-processor
        self.text_processor = TextProcessPipeline(cfg.text_process_pipeline)
        self.text_process_fields = cfg.text_process_fields
        return

    def __iter__(self) -> Iterator[Context]:
        for n, data in enumerate(super().__iter__()):
            # prepare context_id
            if self.id_field is not None:
                context_id = data.pop(self.id_field)
            else:
                context_id = str(n)

            # remove unused fields
            if len(self.saving_fields) > 0:
                data = {key: data.get(key, "") for key in self.saving_fields}

            # preprocess text fields
            for key in self.text_process_fields:
                text = self.text_processor(data[key])
                if text is None:
                    text = ""
                data[key] = text

            formatted_data = Context(context_id=context_id, data=data)
            yield formatted_data
