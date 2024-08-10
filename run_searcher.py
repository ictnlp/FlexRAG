import json
import logging
import os
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from kylin.assistant import Assistant, AssistantConfig
from kylin.metrics import (
    ResponseEvaluator,
    ResponseEvaluatorConfig,
    RetrievalEvaluator,
    RetrievalEvaluatorConfig,
)
from kylin.retriever import RetrievedContext
from kylin.searchers import SearcherConfig, load_searcher
from kylin.utils import COMMIT_ID, SimpleProgressLogger, TimeMeter, read_data

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

    # prepare containers
    contexts: list[list[RetrievedContext]] = []
    tracks = []
    responses = []
    prompts = []
    ckpt_path = os.path.join(
        hydra.utils.get_original_cwd(),
        f"ckpt_{sha256((COMMIT_ID + json.dumps(config)).encode()).hexdigest()}.json",
    )
    if os.path.exists(ckpt_path):
        state = json.load(open(ckpt_path, "r"))
        contexts = state["contexts"]
        tracks = state["search_trackback"]
        responses = state["responses"]
        prompts = state["response_prompts"]
        questions_ = questions[len(responses) :]
    else:
        questions_ = questions

    # search and generate
    p_logger = SimpleProgressLogger(logger, len(questions_), config.log_interval)
    for q in questions_:
        p_logger.update(desc="Searching")
        try:
            # search
            if searcher is not None:
                ctxs, track = searcher.search(q)
                contexts.append(ctxs)
                tracks.append(track)
            else:
                ctxs = []
            # generate
            r, prompt = assistant.answer(question=q, contexts=ctxs)
        # save running state and raise error
        except Exception as e:
            logger.error(f"Error when processing question: {q}")
            with open(ckpt_path, "w") as f:
                json.dump(
                    {
                        "contexts": contexts,
                        "search_trackback": tracks,
                        "responses": responses,
                        "response_prompts": prompts,
                    },
                    f,
                    indent=4,
                )
            raise e
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
        "commit_id": COMMIT_ID,
        "config": config,
        "questions": questions,
        "golden_answers": goldens,
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
        if not os.path.exists(os.path.dirname(data_cfg.output_path)):
            os.makedirs(os.path.dirname(data_cfg.output_path))
        with open(data_cfg.output_path, "w") as f:
            json.dump(final, f, indent=4, ensure_ascii=False)
    return


if __name__ == "__main__":
    main()
