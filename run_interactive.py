import logging
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from kylin.assistant import Assistant, AssistantConfig
from kylin.searchers import load_searcher, SearcherConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class Config(SearcherConfig):
    assistant_config: AssistantConfig = field(default_factory=AssistantConfig)


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: Config):
    # merge config
    default_cfg = OmegaConf.structured(Config)
    config = OmegaConf.merge(default_cfg, config)
    logging.info(f"Configs:\n{OmegaConf.to_yaml(config)}")

    # load searcher
    searcher = load_searcher(config)

    # load assistant
    assistant = Assistant(config.assistant_config)

    while True:
        query = input("Ask me anything(type `quit` to quit): ")
        if query == "quit":
            break
        # search
        if searcher is not None:
            ctxs, _ = searcher.search(query)
        else:
            ctxs = []
        # generate
        r, _ = assistant.answer(questions=[query], contexts=[ctxs])
        print(f"Response: {r[0]}")

    if searcher is not None:
        searcher.close()
    return


if __name__ == "__main__":
    main()
