import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from kylin.models import GenerationConfig, GeneratorConfig, load_generator
from kylin.retriever import RetrievedContext, Retriever
from kylin.utils import Register

logger = logging.getLogger(__name__)


Searchers = Register("searchers")


@dataclass
class BaseSearcherConfig(GeneratorConfig):
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)


class BaseSearcher(ABC):
    retriever: Retriever

    def __init__(self, cfg: BaseSearcherConfig) -> None:
        self.agent = load_generator(cfg)
        self.gen_cfg = cfg.generation_config
        if self.gen_cfg.sample_num > 1:
            logger.warning("Sample num > 1 is not supported for Searcher")
            self.gen_cfg.sample_num = 1
        return

    @abstractmethod
    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[dict[str, object]]]:
        return

    def close(self) -> None:
        self.retriever.close()
        return


@dataclass
class SearchHistory:
    query: str
    contexts: list[RetrievedContext]
