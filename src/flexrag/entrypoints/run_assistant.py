import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from flexrag.assistant import ASSISTANTS
from flexrag.common_dataclass import RetrievedContext
from flexrag.datasets import RAGEvalDataset, RAGEvalDatasetConfig
from flexrag.metrics import Evaluator, EvaluatorConfig
from flexrag.utils import LOGGER_MANAGER, SimpleProgressLogger, load_user_module

# load user modules before loading config
for arg in sys.argv:
    if arg.startswith("user_module="):
        load_user_module(arg.split("=")[1])
        sys.argv.remove(arg)


AssistantConfig = ASSISTANTS.make_config()


@dataclass
class Config(AssistantConfig, RAGEvalDatasetConfig):
    eval_config: EvaluatorConfig = field(default_factory=EvaluatorConfig)  # fmt: skip
    log_interval: int = 10
    output_path: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="default", node=Config)
logger = LOGGER_MANAGER.get_logger("run_assistant")


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: Config):
    # merge config
    default_cfg = OmegaConf.structured(Config)
    config = OmegaConf.merge(default_cfg, config)

    # load dataset
    testset = RAGEvalDataset(config)

    # load assistant
    assistant = ASSISTANTS.load(config)

    # prepare output paths
    if config.output_path is not None:
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
        details_path = os.path.join(config.output_path, "details.jsonl")
        eval_score_path = os.path.join(config.output_path, "eval_score.json")
        config_path = os.path.join(config.output_path, "config.yaml")
        log_path = os.path.join(config.output_path, "log.txt")
    else:
        details_path = os.devnull
        eval_score_path = os.devnull
        config_path = os.devnull
        log_path = os.devnull

    # save config and set logger
    with open(config_path, "w", encoding="utf-8") as f:
        OmegaConf.save(config, f)
    handler = logging.FileHandler(log_path)
    LOGGER_MANAGER.add_handler(handler)
    logger.debug(f"Configs:\n{OmegaConf.to_yaml(config)}")

    # search and generate
    p_logger = SimpleProgressLogger(logger, interval=config.log_interval)
    questions = []
    golden_answers = []
    golden_contexts = []
    responses = []
    contexts: list[list[RetrievedContext]] = []
    with open(details_path, "w", encoding="utf-8") as f:
        for item in testset:
            questions.append(item.question)
            golden_answers.append(item.golden_answers)
            golden_contexts.append(item.golden_contexts)
            response, ctxs, metadata = assistant.answer(question=item.question)
            responses.append(response)
            contexts.append(ctxs)
            json.dump(
                {
                    "question": item.question,
                    "golden": item.golden_answers,
                    "golden_contexts": item.golden_contexts,
                    "metadata_test": item.meta_data,
                    "response": response,
                    "contexts": ctxs,
                    "metadata": metadata,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
            p_logger.update(desc="Searching")

    # evaluate
    evaluator = Evaluator(config.eval_config)
    resp_score, resp_score_detail = evaluator.evaluate(
        questions=questions,
        responses=responses,
        golden_responses=golden_answers,
        retrieved_contexts=contexts,
        golden_contexts=golden_contexts,
        log=True,
    )
    with open(eval_score_path, "w", encoding="utf-8") as f:
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
