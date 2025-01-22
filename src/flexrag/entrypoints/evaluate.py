import json
from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from flexrag.datasets import LineDelimitedDataset, LineDelimitedDatasetConfig
from flexrag.metrics import Evaluator, EvaluatorConfig
from flexrag.utils import LOGGER_MANAGER


@dataclass
class Config(EvaluatorConfig, LineDelimitedDatasetConfig):
    output_path: Optional[str] = None


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
    dataset = LineDelimitedDataset(config)

    questions = [i["question"] for i in dataset]
    responses = [i["response"] for i in dataset]
    golden_answers = [i["golden"] for i in dataset]
    contexts = [i["contexts"] for i in dataset]
    golden_contexts = [i["golden_contexts"] for i in dataset]

    # evaluate
    evaluator = Evaluator(config)
    resp_score, resp_score_detail = evaluator.evaluate(
        questions=questions,
        responses=responses,
        golden_responses=golden_answers,
        retrieved_contexts=contexts,
        golden_contexts=golden_contexts,
        log=True,
    )
    if config.output_path is not None:
        with open(config.output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "eval_scores": resp_score,
                    "eval_details": resp_score_detail,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
    return


if __name__ == "__main__":
    main()
