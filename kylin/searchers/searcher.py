import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from kylin.models import GenerationConfig, GeneratorConfig, load_generator
from kylin.retriever import RetrievedContext, Retriever
from kylin.utils import Register

logger = logging.getLogger(__name__)


Searchers = Register("searchers")


@dataclass
class BaseSearcherConfig:
    retriever_top_k: int = 10
    disable_cache: bool = False


class BaseSearcher(ABC):
    retriever: Retriever

    def __init__(self, cfg: BaseSearcherConfig) -> None:
        self.retriever_top_k = cfg.retriever_top_k
        self.disable_cache = cfg.disable_cache
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
class AgentSearcherConfig(BaseSearcherConfig):
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    generator_config: GeneratorConfig = field(default_factory=GeneratorConfig)  # type: ignore


class AgentSearcher(BaseSearcher):
    def __init__(self, cfg: AgentSearcherConfig) -> None:
        super().__init__(cfg)
        self.agent = load_generator(cfg.generator_config)
        self.gen_cfg = cfg.generation_config
        if self.gen_cfg.sample_num > 1:
            logger.warning("sample_num > 1 is not supported by the AgentSearcher")
            logger.warning("Setting the sample_num to 1")
            self.gen_cfg.sample_num = 1
        return


@dataclass
class SearchHistory:
    query: str
    contexts: list[RetrievedContext]

    def to_dict(self) -> dict[str, str | list[dict]]:
        return {
            "query": self.query,
            "contexts": [ctx.to_dict() for ctx in self.contexts],
        }
