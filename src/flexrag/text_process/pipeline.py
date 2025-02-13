from flexrag.utils import TIME_METER

from .processor import PROCESSORS, Processor, TextUnit


TextProcessPipelineConfig = PROCESSORS.make_config(
    allow_multiple=True, config_name="TextProcessPipelineConfig"
)


class TextProcessPipeline:
    def __init__(self, cfg: TextProcessPipelineConfig) -> None:  # type: ignore
        # load processors
        self.processors: list[Processor] = PROCESSORS.load(cfg)
        return

    @TIME_METER("text_process_pipeline")
    def __call__(self, text: str, return_detail: bool = False) -> str | TextUnit | None:
        unit = TextUnit(content=text)
        for processor in self.processors:
            unit = processor(unit)
            if not unit.reserved:
                break
        if return_detail:
            return unit
        return unit.content if unit.reserved else None

    def __contains__(self, processor: Processor | str) -> bool:
        if isinstance(processor, str):
            return any(
                isinstance(p, PROCESSORS[processor]["item"]) for p in self.processors
            )
        return processor in self.processors

    def __getitem__(self, processor: str | int) -> Processor:
        if isinstance(processor, int):
            return self.processors[processor]
        assert isinstance(processor, str), "str or int is required"
        for p in self.processors:
            if isinstance(p, PROCESSORS[processor]["item"]):
                return p
        raise KeyError(f"Processor {processor} not found in the pipeline")

    def __repr__(self) -> str:
        return f"Pipeline({[p.name for p in self.processors]})"
