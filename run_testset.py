import json
import os
import logging
from argparse import ArgumentParser

from kylin.kylin_searcher import KylinLLMSearcher
from kylin.metrics import ShortFormEvaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # load arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The path to the dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to dump the results",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="The interval to log the progress",
    )
    parser = KylinLLMSearcher.add_args(parser)
    parser = ShortFormEvaluator.add_args(parser)
    args = parser.parse_args()

    # load dataset
    testdata = [json.loads(i) for i in open(args.data_path, "r")]
    questions = [i["question"] for i in testdata]
    goldens = [i["golden_answers"] for i in testdata]

    # search
    searcher = KylinLLMSearcher(args)
    responses = []
    responses, tracks = searcher.answer(questions)
    searcher.close()

    # evaluate
    evaluator = ShortFormEvaluator(args)
    r, r_detail = evaluator.evaluate(goldens, responses)
    final = {
        "responses": responses,
        "search_trackback": tracks,
        "scores": r,
        "score_details": r_detail,
    }

    # dump results
    with open(args.output_path, "w") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
