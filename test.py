from argparse import ArgumentParser
from retriever import DenseRetriever


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="The interval to log the progress",
    )
    parser = DenseRetriever.add_args(parser)
    args = parser.parse_args()
    retriever = DenseRetriever(args)
    while True:
        query = input("Query: ")
        if query == "quit":
            break
        r = retriever.search([query], leaves_to_search=500)[0]
        pass
    retriever.test_speed(sample_num=8192, top_k=5, leaves_to_search=500)
    retriever.close()
    pass
