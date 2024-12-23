import json
from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from librarian.data import LineDelimitedDataset
from librarian.metrics import RAGEvaluatorConfig, RAGEvaluator
from librarian.utils import LOGGER_MANAGER


@dataclass
class Config(RAGEvaluatorConfig):
    data_path: str = MISSING
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
    dataset = LineDelimitedDataset(config.data_path)

    questions = [i["question"] for i in dataset]
    responses = [i["response"] for i in dataset]
    golden_answers = [i["golden"] for i in dataset]
    contexts = [i["contexts"] for i in dataset]
    golden_contexts = [i["golden_contexts"] for i in dataset]

    # evaluate
    evaluator = RAGEvaluator(config)
    resp_score, resp_score_detail = evaluator.evaluate(
        questions=questions,
        responses=responses,
        golden_responses=golden_answers,
        retrieved_contexts=contexts,
        golden_contexts=golden_contexts,
        log=True,
    )
    if config.output_path is not None:
        with open(config.output_path, "w") as f:
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
