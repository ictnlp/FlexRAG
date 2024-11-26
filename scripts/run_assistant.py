import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from kylin.assistant import ASSISTANTS
from kylin.metrics import (
    ResponseEvaluator,
    ResponseEvaluatorConfig,
    RetrievalEvaluator,
    RetrievalEvaluatorConfig,
)
from kylin.retriever import RetrievedContext
from kylin.utils import (
    LOGGER_MANAGER,
    SimpleProgressLogger,
    read_data,
    load_user_module,
)

# load user modules before loading config
for arg in sys.argv:
    if arg.startswith("user_module="):
        load_user_module(arg.split("=")[1])
        sys.argv.remove(arg)


@dataclass
class DataConfig:
    data_path: str = MISSING
    data_range: Optional[list[int]] = None
    output_path: Optional[str] = None


AssistantConfig = ASSISTANTS.make_config()


@dataclass
class Config(AssistantConfig, DataConfig):
    retrieval_eval_config: RetrievalEvaluatorConfig = field(default_factory=RetrievalEvaluatorConfig)  # fmt: skip
    response_eval_config: ResponseEvaluatorConfig = field(default_factory=ResponseEvaluatorConfig)  # fmt: skip
    log_interval: int = 10


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: Config):
    # merge config
    default_cfg = OmegaConf.structured(Config)
    config = OmegaConf.merge(default_cfg, config)

    # load dataset
    testdata = read_data(config.data_path, config.data_range)

    # load assistant
    assistant = ASSISTANTS.load(config)

    # prepare output paths
    if config.output_path is not None:
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
        details_path = os.path.join(config.output_path, "details.jsonl")
        retrieval_score_path = os.path.join(config.output_path, "retrieval_score.json")
        response_score_path = os.path.join(config.output_path, "response_score.json")
        config_path = os.path.join(config.output_path, "config.yaml")
        log_path = os.path.join(config.output_path, "log.txt")
    else:
        details_path = "/dev/null"
        retrieval_score_path = "/dev/null"
        response_score_path = "/dev/null"
        config_path = "/dev/null"
        log_path = "/dev/null"

    # save config and set logger
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
    handler = logging.FileHandler(log_path)
    logger = LOGGER_MANAGER.get_logger("run_assistant")
    LOGGER_MANAGER.add_handler(handler)
    logger.debug(f"Configs:\n{OmegaConf.to_yaml(config)}")

    # search and generate
    p_logger = SimpleProgressLogger(logger, interval=config.log_interval)
    questions = []
    goldens = []
    responses = []
    contexts: list[list[RetrievedContext]] = []
    with open(details_path, "w") as f:
        for item in testdata:
            questions.append(item["question"])
            goldens.append(item["golden_answers"])
            response, ctxs, metadata = assistant.answer(question=item["question"])
            responses.append(response)
            contexts.append(ctxs)
            json.dump(
                {
                    "question": item["question"],
                    "golden": item["golden_answers"],
                    "response": response,
                    "contexts": ctxs,
                    "metadata": metadata,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
            p_logger.update(desc="Searching")

    # evaluate retrieval
    if contexts[0] is not None:
        evaluator = RetrievalEvaluator(config.retrieval_eval_config)
        ret_score, ret_score_detail = evaluator.evaluate(goldens, contexts)
        with open(retrieval_score_path, "w") as f:
            json.dump(
                {
                    "retrieval_scores": ret_score,
                    "retrieval_scores_details": ret_score_detail,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
    else:
        ret_score, ret_score_detail = None, None

    # evaluate response
    evaluator = ResponseEvaluator(config.response_eval_config)
    resp_score, resp_score_detail = evaluator.evaluate(goldens, responses)
    with open(response_score_path, "w") as f:
        json.dump(
            {
                "response_scores": resp_score,
                "response_scores_details": resp_score_detail,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
    return


if __name__ == "__main__":
    main()
