import json
import os
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from flexrag.metrics import Evaluator
from flexrag.utils import LOGGER_MANAGER


@dataclass
class Config:
    result_paths: list[str] = MISSING
    output_path: str = MISSING


cs = ConfigStore.instance()
cs.store(name="default", node=Config)
logger = LOGGER_MANAGER.get_logger("combine_outputs")


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config):
    # merge config
    default_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(default_cfg, cfg)

    # load the metadata
    config_path = os.path.join(cfg.result_paths[0], "config.yaml")
    loaded_config = OmegaConf.load(config_path)
    evaluator = Evaluator(loaded_config.eval_config)

    # prepare output path
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)
    output_details_path = os.path.join(cfg.output_path, "details.jsonl")
    output_eval_score_path = os.path.join(cfg.output_path, "eval_score.json")
    output_config_path = os.path.join(cfg.output_path, "config.yaml")
    OmegaConf.save(loaded_config, output_config_path)

    # combine the results
    logger.info("Combining the results...")
    questions = []
    golden_answers = []
    golden_contexts = []
    responses = []
    contexts = []
    with open(output_details_path, "w", encoding="utf-8") as f:
        for result_path in cfg.result_paths:
            details_path = os.path.join(result_path, "details.jsonl")
            for line in open(details_path, "r", encoding="utf-8"):
                f.write(line)
                data = json.loads(line)
                questions.append(data["question"])
                golden_answers.append(data["golden"])
                golden_contexts.append(data["golden_contexts"])
                responses.append(data["response"])
                contexts.append(data["contexts"])

    # re-evaluate the combined results
    logger.info("Re-evaluating the combined results...")
    resp_score, resp_score_detail = evaluator.evaluate(
        questions=questions,
        responses=responses,
        golden_responses=golden_answers,
        retrieved_contexts=contexts,
        golden_contexts=golden_contexts,
        log=True,
    )
    with open(output_eval_score_path, "w", encoding="utf-8") as f:
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
