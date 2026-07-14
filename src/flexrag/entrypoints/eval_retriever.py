from __future__ import annotations

import logging
import os
import sys
from dataclasses import field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore

from flexrag.common import (
    LOGGER_MANAGER,
    SimpleProgressLogger,
    configure,
    extract_config,
    load_user_module,
)
from flexrag.common.dataclasses import Context, RetrievedContext
from flexrag.common.serialization import json_dump
from flexrag.datasets import MTEBDataset, MTEBDatasetConfig
from flexrag.metrics import Evaluator, EvaluatorConfig
from flexrag.retrievers import FlexRetrieverConfig

from ._retriever_builder import (
    CollectionBackendInitConfig,
    ContextStoreInitConfig,
    build_flex_retriever,
)

for arg in list(sys.argv):
    if arg.startswith("user_module="):
        load_user_module(arg.split("=")[1])
        sys.argv.remove(arg)


@configure
class Config(MTEBDatasetConfig):
    """Configuration for evaluating one explicitly constructed retriever."""

    output_path: Optional[str] = None
    eval_config: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    log_interval: int = 10
    context_store: ContextStoreInitConfig = field(
        default_factory=ContextStoreInitConfig
    )
    backend: CollectionBackendInitConfig = field(
        default_factory=CollectionBackendInitConfig
    )
    retriever_config: FlexRetrieverConfig = field(default_factory=FlexRetrieverConfig)


cs = ConfigStore.instance()
cs.store(name="default", node=Config)
logger = LOGGER_MANAGER.get_logger("run_retriever")


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: Config) -> None:
    config = extract_config(config, Config)
    testset = MTEBDataset(config)
    retriever = build_flex_retriever(
        backend_config=config.backend,
        context_store_config=config.context_store,
        retriever_config=config.retriever_config,
    )

    if config.output_path is not None:
        os.makedirs(config.output_path, exist_ok=True)
        details_path = os.path.join(config.output_path, "details.jsonl")
        eval_score_path = os.path.join(config.output_path, "eval_score.json")
        config_path = os.path.join(config.output_path, "config.yaml")
        log_path = os.path.join(config.output_path, "log.txt")
    else:
        details_path = os.devnull
        eval_score_path = os.devnull
        config_path = os.devnull
        log_path = os.devnull

    config.dump(config_path)
    handler = logging.FileHandler(log_path)
    LOGGER_MANAGER.add_handler(handler)
    logger.debug(f"Configs:\n{config.dumps()}")

    try:
        p_logger = SimpleProgressLogger(logger, interval=config.log_interval)
        questions = []
        goldens: list[list[Context]] = []
        retrieved: list[list[RetrievedContext]] = []
        with open(details_path, "w", encoding="utf-8") as f:
            for item in testset:
                questions.append(item.question)
                goldens.append(item.contexts)
                ctxs = retriever.search(queries=item.question)[0]
                retrieved.append(ctxs)
                f.write(
                    json_dump(
                        {
                            "question": item.question,
                            "golden_contexts": item.contexts,
                            "metadata": item.metadata,
                            "contexts": ctxs,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                p_logger.update(desc="Searching")

        evaluator = Evaluator(config.eval_config)
        resp_score, resp_score_detail = evaluator.evaluate(
            questions=questions,
            retrieved_contexts=retrieved,
            golden_contexts=goldens,
            log=True,
        )
        with open(eval_score_path, "w", encoding="utf-8") as f:
            f.write(
                json_dump(
                    {
                        "eval_scores": resp_score,
                        "eval_details": resp_score_detail,
                    },
                    indent=4,
                    ensure_ascii=False,
                )
            )
    finally:
        for backend in retriever.backends.values():
            backend.close()
        if retriever.context_store is not None:
            retriever.context_store.close()
    return


if __name__ == "__main__":
    main()
