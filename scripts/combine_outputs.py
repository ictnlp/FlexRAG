import json
import pathlib
import sys
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf, DictConfig

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from kylin.metrics import ResponseEvaluator, RetrievalEvaluator


@dataclass
class Config:
    result_paths: list[str] = MISSING
    output_path: str = MISSING


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config):
    default_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(default_cfg, cfg)
    results = [json.load(open(p, "r")) for p in cfg.result_paths]

    # combine the results
    questions = []
    goldens = []
    contexts = []
    responses = []
    response_prompts = []
    search_trackback = []
    for r in results:
        questions.extend(r["questions"])
        goldens.extend(r["golden_answers"])
        contexts.extend(r["contexts"])
        responses.extend(r["responses"])
        response_prompts.extend(r["response_prompts"])
        search_trackback.extend(r["search_trackback"])
    assert len(questions) == len(goldens)
    if len(contexts) > 0:
        assert len(questions) == len(contexts)
        assert len(questions) == len(search_trackback)
    if len(responses) > 0:
        assert len(questions) == len(responses)
        assert len(questions) == len(response_prompts)

    # combine the meters
    time_meters = [i["time_meter"] for i in results]
    time_meter = []
    for items in zip(*time_meters):
        total_time = sum([i["total time"] for i in items])
        total_calls = sum([i["calls"] for i in items])
        time_meter.append(
            {
                "name": items[0]["name"],
                "calls": total_calls,
                "average call time": total_time / total_calls,
                "total time": total_time,
            }
        )

    # load evaluator
    res_eval_configs = [i["config"]["response_eval_config"] for i in results]
    ret_eval_configs = [i["config"]["retrieval_eval_config"] for i in results]
    assert all([i == res_eval_configs[0] for i in res_eval_configs])
    assert all([i == ret_eval_configs[0] for i in ret_eval_configs])
    res_eval_cfg = DictConfig(results[0]["config"]["response_eval_config"])
    ret_eval_cfg = DictConfig(results[0]["config"]["retrieval_eval_config"])
    res_eval = ResponseEvaluator(res_eval_cfg)
    ret_eval = RetrievalEvaluator(ret_eval_cfg)

    # re-evaluate
    if len(responses) > 0:
        resp_score, resp_score_detail = res_eval.evaluate(goldens, responses)
    else:
        resp_score, resp_score_detail = None, None
    if len(contexts) > 0:
        contexts_text = [[i["full_text"] for i in ctx] for ctx in contexts]
        ret_score, ret_score_detail = ret_eval.evaluate(goldens, contexts_text)
    else:
        ret_score, ret_score_detail = None, None

    # save the combined results
    combined_results = {
        "commit_id": None,
        "config": None,
        "questions": questions,
        "golden_answers": goldens,
        "contexts": contexts,
        "responses": responses,
        "response_prompts": response_prompts,
        "search_trackback": search_trackback,
        "retrieval_scores": ret_score,
        "retrieval_scores_details": ret_score_detail,
        "response_scores": resp_score,
        "response_scores_details": resp_score_detail,
        "time_meter": time_meter,
    }
    with open(cfg.output_path, "w") as f:
        json.dump(combined_results, f, indent=4, ensure_ascii=False)
    return


if __name__ == "__main__":
    main()
