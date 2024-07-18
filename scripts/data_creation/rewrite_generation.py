import json
import logging
import pathlib
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


from kylin.kylin_prompts import rewrite_prompts
from kylin.metrics import RetrievalEvaluator, RetrievalEvaluatorConfig
from kylin.models import GenerationConfig, OpenAIGenerator, OpenAIGeneratorConfig
from kylin.retriever import BM25Retriever, BM25RetrieverConfig
from kylin.utils import SimpleProgressLogger, read_data


@dataclass
class DataConfig:
    data_path: list[str] = field(default_factory=list)
    data_range: Optional[list[list[int]]] = field(default=None)
    output_path: str = MISSING


@dataclass
class Config:
    data_config: DataConfig = field(default_factory=DataConfig)
    retriever_config: BM25RetrieverConfig = field(default_factory=BM25RetrieverConfig)
    generator_config: OpenAIGeneratorConfig = field(default_factory=OpenAIGeneratorConfig)  # fmt: skip
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)  # fmt: skip
    eval_config: RetrievalEvaluatorConfig = field(default_factory=RetrievalEvaluatorConfig)  # fmt: skip
    disable_cache: bool = False
    log_interval: int = 10


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


def rewrite_query(
    info: str,
    generator: OpenAIGenerator,
    gen_cfg: GenerationConfig,
) -> list[str]:
    # Rewrite the query to be more informative
    user_prompt = f"Query: {info}"
    prompt = deepcopy(rewrite_prompts["bm25"])
    prompt.append({"role": "user", "content": user_prompt})
    queries = generator.chat([prompt], generation_config=gen_cfg)[0]
    return queries


@hydra.main(version_base="1.1", config_path=None, config_name="default")
def main(config: Config):
    # merge config
    default_cfg = OmegaConf.structured(Config)
    config = OmegaConf.merge(default_cfg, config)
    logging.info(f"Configs:\n{OmegaConf.to_yaml(config)}")

    # load dataset
    datasets = read_data(config.data_config.data_path, config.data_config.data_range)

    # load retriever
    retriever = BM25Retriever(config.retriever_config)

    # load generator
    generator = OpenAIGenerator(config.generator_config)

    # load evaluator
    evaluator = RetrievalEvaluator(config.eval_config)

    # generate
    p_logger = SimpleProgressLogger(logger, None, config.log_interval)
    results = []
    for datapoint in datasets:
        question = datapoint["question"]
        golden = datapoint["golden_answers"]
        results.append(
            {
                "question": question,
                "rewrite_queries": [],
            }
        )
        p_logger.update(desc="Rewriting")

        # rewrite the query
        while True:
            rwt_queries = rewrite_query(question, generator, config.generation_config)
            rwt_queries.append(question)
            # filter out noisy queries
            rwt_queries = list(set(rwt_queries))
            rwt_queries = list(
                filter(lambda x: len(x) / len(question) < 5, rwt_queries)
            )
            if len(rwt_queries) > 1:
                break

        # search
        contexts = retriever.search(
            query=rwt_queries,
            disable_cache=config.disable_cache,
        )

        # evaluate
        for q, ctxs in zip(rwt_queries, contexts):
            ctxs = [i["text"] for i in ctxs]
            r, _ = evaluator.evaluate([golden], [ctxs], log=False)
            results[-1]["rewrite_queries"].append(
                {
                    "query": q,
                    "ctxs": ctxs,
                    "score": r,
                }
            )

    # dump the results
    with open(config.data_config.output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
