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
    ShortFormEvaluator,
    ShortFormEvaluatorConfig,
    LongFormEvaluator,
    LongFormEvaluatorConfig,
)
from kylin.searchers import (
    BM25Searcher,
    BM25SearcherConfig,
    WebSearcher,
    WebSearcherConfig,
)
from kylin.utils import SimpleProgressLogger, Choices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    data_path: str = MISSING
    output_path: str = MISSING


@dataclass
class Config:
    data_config: DataConfig = field(default_factory=DataConfig)
    assistant_config: AssistantConfig = field(default_factory=AssistantConfig)
    searcher_type: Choices(["bm25", "web"]) = "bm25"  # type: ignore
    bm25_searcher_config: BM25SearcherConfig = field(default_factory=BM25SearcherConfig)
    web_searcher_config: WebSearcherConfig = field(default_factory=WebSearcherConfig)
    response_type: Optional[Choices(["short", "long"])] = None  # type: ignore
    retrieval_eval_config: RetrievalEvaluatorConfig = field(default_factory=RetrievalEvaluatorConfig)  # fmt: skip
    short_eval_config: ShortFormEvaluatorConfig = field(default_factory=ShortFormEvaluatorConfig)  # fmt: skip
    long_eval_config: LongFormEvaluatorConfig = field(default_factory=LongFormEvaluatorConfig)  # fmt: skip
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
    testdata = [json.loads(i) for i in open(data_cfg.data_path, "r")]
    questions = [i["question"] for i in testdata]
    goldens = [i["golden_answers"] for i in testdata]

    # load searcher
    match config.searcher_type:
        case "bm25":
            searcher = BM25Searcher(config.bm25_searcher_config)
        case "web":
            searcher = WebSearcher(config.web_searcher_config)
        case _:
            raise ValueError(f"Invalid searcher type: {config.searcher_type}")

    # load assistant
    if config.response_type is not None:
        assistant = Assistant(config.assistant_config)
    else:
        assistant = None

    contexts = []
    tracks = []
    responses = []
    prompts = []
    pbar = SimpleProgressLogger(logger, len(questions), config.log_interval)
    for q in questions:
        # search
        pbar.update(desc="Searching")
        ctxs, track = searcher.search(q)
        contexts.append(ctxs)
        tracks.append(track)

        # generate
        if assistant is not None:
            r, prompt = assistant.answer(question=q, contexts=ctxs)
            responses.append(r)
            prompts.append(prompt)
    searcher.close()

    # evaluate retrieval
    contexts_text = [[i["full_text"] for i in ctx] for ctx in contexts]
    evaluator = RetrievalEvaluator(config.retrieval_eval_config)
    ret_score, ret_score_detail = evaluator.evaluate(goldens, contexts_text)

    # evaluate
    match config.response_type:
        case "long":
            evaluator = LongFormEvaluator(config.long_eval_config)
            resp_score, resp_score_detail = evaluator.evaluate(goldens, responses)
        case "short":
            evaluator = ShortFormEvaluator(config.short_eval_config)
            resp_score, resp_score_detail = evaluator.evaluate(goldens, responses)
        case None:
            resp_score, resp_score_detail = None, None

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
    }
    with open(data_cfg.output_path, "w") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    return


if __name__ == "__main__":
    main()
