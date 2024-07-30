import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from kylin.assistant import Assistant, AssistantConfig
from kylin.metrics import (
    RetrievalEvaluator,
    RetrievalEvaluatorConfig,
    ResponseEvaluator,
    ResponseEvaluatorConfig,
)
from kylin.searchers import SearcherConfig, load_searcher
from kylin.retriever import RetrievedContext
from kylin.utils import SimpleProgressLogger, read_data, TimeMeter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    data_path: str = MISSING
    data_range: Optional[list[int]] = None
    output_path: Optional[str] = None


@dataclass
class Config(SearcherConfig):
    data_config: DataConfig = field(default_factory=DataConfig)
    assistant_config: AssistantConfig = field(default_factory=AssistantConfig)
    retrieval_eval_config: RetrievalEvaluatorConfig = field(default_factory=RetrievalEvaluatorConfig)  # fmt: skip
    response_eval_config: ResponseEvaluatorConfig = field(default_factory=ResponseEvaluatorConfig)  # fmt: skip
    log_interval: int = 10


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.1", config_path=None, config_name="default")
def main(config: Config):
    # merge config
    default_cfg = OmegaConf.structured(Config)
    config = OmegaConf.merge(default_cfg, config)
    logging.info(f"Configs:\n{OmegaConf.to_yaml(config)}")

    # load dataset
    data_cfg = config.data_config
    testdata = list(read_data(data_cfg.data_path, data_cfg.data_range))
    questions = [i["question"] for i in testdata]
    goldens = [i["golden_answers"] for i in testdata]

    # load searcher
    searcher = load_searcher(config)

    # load assistant
    assistant = Assistant(config.assistant_config)

    contexts: list[list[RetrievedContext]] = []
    tracks = []
    responses = []
    prompts = []
    p_logger = SimpleProgressLogger(logger, len(questions), config.log_interval)
    for q in questions:
        p_logger.update(desc="Searching")
        # search
        if searcher is not None:
            ctxs, track = searcher.search(q)
            contexts.append(ctxs)
            tracks.append(track)
        else:
            ctxs = []
        # generate
        r, prompt = assistant.answer(question=q, contexts=ctxs)
        responses.append(r)
        prompts.append(prompt)

    if searcher is not None:
        searcher.close()

    # evaluate retrieval
    if searcher is not None:
        contexts_text = [[i.full_text for i in ctx] for ctx in contexts]
        evaluator = RetrievalEvaluator(config.retrieval_eval_config)
        ret_score, ret_score_detail = evaluator.evaluate(goldens, contexts_text)
    else:
        ret_score, ret_score_detail = None, None

    # evaluate response
    evaluator = ResponseEvaluator(config.response_eval_config)
    resp_score, resp_score_detail = evaluator.evaluate(goldens, responses)

    # dump results
    final = {
        "config": config,
        "contexts": contexts,
        "responses": responses,
        "response_prompts": prompts,
        "search_trackback": tracks,
        "retrieval_scores": ret_score,
        "retrieval_scores_details": ret_score_detail,
        "response_scores": resp_score,
        "response_scores_details": resp_score_detail,
        "time_meter": TimeMeter.statistics,
    }
    if data_cfg.output_path is not None:
        with open(data_cfg.output_path, "w") as f:
            json.dump(final, f, indent=4, ensure_ascii=False)
    return


if __name__ == "__main__":
    main()
