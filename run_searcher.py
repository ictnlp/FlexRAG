import json
import logging
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from kylin.metrics import RetrievalEvaluator, RetrievalEvaluatorConfig
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
    searcher_type: Choices(["bm25", "web"]) = "bm25"  # type: ignore
    bm25_searcher_config: BM25SearcherConfig = field(default_factory=BM25SearcherConfig)
    web_searcher_config: WebSearcherConfig = field(default_factory=WebSearcherConfig)
    evaluate_config: RetrievalEvaluatorConfig = field(default_factory=RetrievalEvaluatorConfig)  # fmt: skip
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

    # search
    contexts = []
    tracks = []
    pbar = SimpleProgressLogger(logger, len(questions), config.log_interval)
    for q in questions:
        pbar.update(desc="Searching")
        ctxs, track = searcher.search(q)
        contexts.append(ctxs)
        tracks.append(track)
    searcher.close()

    # evaluate
    contexts_text = [[i["full_text"] for i in ctx] for ctx in contexts]
    evaluator = RetrievalEvaluator(config.evaluate_config)
    r, r_detail = evaluator.evaluate(goldens, contexts_text)

    # dump results
    final = {
        "config": config,
        "contexts": contexts,
        "search_trackback": tracks,
        "scores": r,
        "score_details": r_detail,
    }
    with open(data_cfg.output_path, "w") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    return


if __name__ == "__main__":
    main()
