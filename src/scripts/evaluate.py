from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from librarian.data import IterableDataset
from librarian.metrics import RAGEvaluatorConfig, RAGEvaluator
from librarian.utils import LOGGER_MANAGER


@dataclass
class Config(RAGEvaluatorConfig):
    data_path: str = MISSING


cs = ConfigStore.instance()
cs.store(name="default", node=Config)
logger = LOGGER_MANAGER.get_logger("evaluate")


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: Config):
    # merge config
    default_cfg = OmegaConf.structured(Config)
    config = OmegaConf.merge(default_cfg, config)
    logger.debug(f"Configs:\n{OmegaConf.to_yaml(config)}")

    # load dataset
    dataset = IterableDataset(config.data_path)

    questions = [i["question"] for i in dataset]
    responses = [i["response"] for i in dataset]
    golden_answers = [i["golden"] for i in dataset]
    contexts = [i["contexts"] for i in dataset]
    golden_contexts = [i["golden_contexts"] for i in dataset]

    # evaluate
    evaluator = RAGEvaluator(config)
    evaluator.evaluate(
        questions=questions,
        responses=responses,
        golden_responses=golden_answers,
        retrieved_contexts=contexts,
        golden_contexts=golden_contexts,
        log=True,
    )
    return


if __name__ == "__main__":
    main()
