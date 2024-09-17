from dataclasses import field, make_dataclass
from typing import Optional

from kylin.utils import Choices

from .processor import PROCESSORS, Processor, TextUnit

pipeline_fields = [("processors", Optional[Choices(PROCESSORS.names)], None)]
pipeline_fields += [
    (
        f"{PROCESSORS[name]['short_names'][0]}_config",
        PROCESSORS[name]["config_class"],
        field(default_factory=PROCESSORS[name]["config_class"]),
    )
    for name in PROCESSORS.mainnames
    if PROCESSORS[name]["config_class"] is not None
]
PipelineConfig = make_dataclass("PipelineConfig", pipeline_fields)


class Pipeline:
    def __init__(self, cfg: PipelineConfig) -> None:  # type: ignore
        # load processors
        self.processors: list[Processor] = []
        for name in cfg.processors:
            short_name = PROCESSORS[name]["short_names"][0]
            processor_cfg = getattr(cfg, f"{short_name}_config", None)
            if processor_cfg is not None:
                self.processors.append(PROCESSORS[name]["item"](processor_cfg))
            else:
                self.processors.append(PROCESSORS[name]["item"]())
        return

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
