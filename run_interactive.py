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
    parser = KylinLLMSearcher.add_args(parser)
    parser = ShortFormEvaluator.add_args(parser)
    args = parser.parse_args()

    # search
    searcher = KylinLLMSearcher(args)
    while True:
        query = input("Question:")
        if query == "quit":
            break
        responses, tracks = searcher.answer([query])
        print(f"Answer: {responses[0]}")
    searcher.close()
