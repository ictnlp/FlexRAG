# Benchmark for Single-Hop QA Tasks
To better understand the performance of each component in FlexRAG within the RAG pipeline, we have conducted a series of experiments on a variety of single-hop QA datasets including PopQA, Natural Questions, and TriviaQA. The experiments are divided into four categories: sparse retriever benchmarks, dense retriever benchmarks, index benchmarks, reranker benchmarks, and generator benchmarks. We also provide best practices for single-hop QA tasks.

All experiments are conducted using the `ModularAssistant` in FlexRAG framework.

## Sparse Retriever Benchmarks
> Experiment Settings: In these experiments, we employ the Qwen/Qwen2-7B-Instruct as our generator. All the other settings are the same as the default settings in `ModularAssistant`.

| Methods       | PopQA(%) |       |           | NQ(%) |       |           | TriviaQA(%) |       |           | Average |       |           |
| ------------- | :------: | :---: | :-------: | :---: | :---: | :-------: | :---------: | :---: | :-------: | :-----: | :---: | :-------: |
|               |    F1    |  EM   | Recall@10 |  F1   |  EM   | Recall@10 |     F1      |  EM   | Recall@10 |   F1    |  EM   | Recall@10 |
| BM25s+Lucene  |  57.88   | 52.75 |   68.48   | 38.79 | 30.00 |   54.74   |    65.93    | 58.02 |   61.98   |  54.20  | 46.92 |   61.73   |
| BM25s+BM25+   |  57.88   | 52.75 |   68.48   | 38.79 | 30.00 |   54.74   |    65.92    | 58.01 |   61.99   |  54.20  | 46.92 |   61.74   |
| BM25s+BM25l   |  57.97   | 53.04 |   66.55   | 36.54 | 28.12 |   50.39   |    62.70    | 54.75 |   58.15   |  52.40  | 45.30 |   58.36   |
| BM25s+Atire   |  57.88   | 52.75 |   68.48   | 38.79 | 30.00 |   54.74   |    65.92    | 58.01 |   61.99   |  54.20  | 46.92 |   61.74   |
| ElasticSearch |  57.29   | 52.39 |   66.12   | 36.70 | 28.39 |   52.05   |    65.94    | 58.35 |   62.23   |  53.31  | 46.38 |   60.13   |
| Typesense     |  19.41   | 17.80 |   26.38   | 20.57 | 14.88 |   15.87   |    44.69    | 38.49 |   20.48   |  28.22  | 23.72 |   20.91   |

Observations:
- BM25s+Lucene, BM25s+BM25+, and BM25s+Atire have similar performance on all datasets.
- Typesense struggles to retrieve relevant documents when using natural language queries.
- ElasticSearch has balanced performance across three datasets, and it offers a wide range of retrieval features.

Conclusion:
Considering the excellent performance and simple installation of the BM25S, we believe that using the BM25S as a sparse retriever in research and prototype building scenarios is a better choice. For more complex requirements, try using ElasticSearch, whose out-of-the-box configuration and rich features can provide more powerful retrieval capabilities

## Dense Retriever Benchmarks
> Experiment Settings: In these experiments, we employ the Qwen/Qwen2-7B-Instruct as our generator. All the other settings are the same as the default settings in `ModularAssistant`.

| Methods                                      | PopQA(%) |       |           | NQ(%) |       |           | TriviaQA(%) |       |           | Average |       |           |
| -------------------------------------------- | :------: | :---: | :-------: | :---: | :---: | :-------: | :---------: | :---: | :-------: | :-----: | :---: | :-------: |
|                                              |    F1    |  EM   | Recall@10 |  F1   |  EM   | Recall@10 |     F1      |  EM   | Recall@10 |   F1    |  EM   | Recall@10 |
| facebook/contriever-msmarco                  |  64.14   | 59.04 |   80.77   | 49.67 | 39.03 |   75.65   |    70.36    | 62.55 |   68.26   |  61.39  | 53.54 |   74.89   |
| intfloat/e5-base-v2                          |  59.74   | 54.25 |   77.20   | 50.05 | 39.56 |   78.84   |    71.66    | 63.79 |   70.63   |  60.48  | 52.53 |   75.56   |
| BAAI/bge-m3                                  |  63.65   | 58.76 |   83.42   | 50.98 | 40.36 |   80.00   |    71.92    | 63.85 |   71.10   |  62.18  | 54.32 |   78.17   |
| sentence-transformers/msmarco-MiniLM-L-12-v3 |  64.76   | 59.11 |   80.84   | 42.78 | 33.77 |   64.60   |    58.10    | 50.72 |   51.77   |  55.21  | 47.87 |   65.74   |
| nomic-ai/nomic-embed-text-v1.5               |  65.06   | 59.90 |   81.70   | 50.31 | 40.08 |   78.14   |    69.10    | 61.32 |   67.50   |  61.49  | 53.77 |   75.78   |
| jinaai/jina-embeddings-v3                    |  67.43   | 62.33 |   86.20   | 50.02 | 40.17 |   81.52   |    70.06    | 62.14 |   79.51   |  62.50  | 54.88 |   82.41   |
| facebook/dragon-plus-query-encoder           |  66.67   | 61.69 |   84.06   | 46.79 | 37.17 |   73.80   |    70.30    | 62.54 |   68.40   |  61.25  | 53.80 |   75.42   |

Observations:
- All dense retrievers have better performance than sparse retrievers.
- jina-embeddings-v3 and BGE M3 have the best performance on all datasets.
- MiniLM provides a balance choice between performance and efficiency.

Conclusion:
We recommend using facebook/contriever-msmarco or E5 for academic usage as it is used in many papers and has a good balance between performance and efficiency. For building a prototype or production system, we recommend using jina-embeddings-v3 or BGE M3, which have the best performance on all datasets.


## Index Benchmarks
> Experiment Settings: In these experiments, we employ the Qwen/Qwen2-7B-Instruct as our generator and facebook/contriever-msmarco as our dense retriever. All the other settings are the same as the default settings in `ModularAssistant`.

| Methods                | PopQA(%) |       |       | NQ(%) |       |       | TriviaQA(%) |       |       | Average |       |       |
| ---------------------- | :------: | :---: | :---: | :---: | :---: | :---: | :---------: | :---: | :---: | :-----: | :---: | :---: |
|                        |    F1    |  EM   | Succ  |  F1   |  EM   | Succ  |     F1      |  EM   | Succ  |   F1    |  EM   | Succ  |
| FLAT                   |  63.65   | 58.40 | 82.20 | 49.20 | 39.11 | 77.95 |    70.61    | 62.70 | 80.03 |  61.15  | 53.40 | 80.06 |
| Faiss Auto(nprobe=32)  |  51.50   | 47.03 | 67.19 | 48.17 | 37.89 | 75.21 |    69.34    | 61.56 | 78.36 |  56.34  | 48.83 | 73.59 |
| Faiss Auto(nprobe=128) |  59.91   | 54.97 | 76.20 | 49.05 | 38.53 | 77.23 |    70.14    | 62.31 | 79.49 |  59.70  | 51.94 | 77.64 |
| Faiss Auto(nprobe=512) |  64.14   | 59.04 | 81.42 | 49.62 | 39.11 | 77.87 |    70.48    | 62.57 | 79.80 |  61.41  | 53.57 | 79.70 |
| Faiss Refine           |  64.11   | 58.90 | 81.27 | 48.91 | 38.34 | 77.81 |    70.24    | 62.43 | 79.89 |  61.09  | 53.22 | 79.66 |
| ScaNN                  |  63.26   | 58.11 | 82.13 | 49.31 | 39.25 | 77.76 |    70.50    | 62.64 | 79.93 |  61.02  | 53.33 | 79.94 |


Observations:
- Faiss provides a good balance between performance and efficiency.
- ScaNN offers high retrieval speed and accuracy, but it consumes a large amount of memory, making it suitable for use on platforms with ample memory.


## Reranker Benchmarks
> Experiment Settings: In these experiments, we employ the Qwen/Qwen2-7B-Instruct, facebook/contriever-msmarco, and Faiss Auto(nprobe=512) as our generator, dense retriever, and index, respectively. All the other settings are the same as the default settings in `ModularAssistant`.

| Methods                                   | PopQA(%) |       |       | NQ(%) |       |       | TriviaQA(%) |       |       | Average |       |       |
| ----------------------------------------- | :------: | :---: | :---: | :---: | :---: | :---: | :---------: | :---: | :---: | :-----: | :---: | :---: |
|                                           |    F1    |  EM   | Succ  |  F1   |  EM   | Succ  |     F1      |  EM   | Succ  |   F1    |  EM   | Succ  |
| BAAI/bge-reranker-v2-m3                   |  66.02   | 60.76 | 86.92 | 50.94 | 40.53 | 81.91 |    74.58    | 66.71 | 84.81 |  63.85  | 56.00 | 84.55 |
| colbert-ir/colbertv2.0                    |  65.44   | 60.47 | 83.56 | 47.18 | 37.06 | 77.53 |    72.13    | 64.24 | 81.47 |  61.58  | 53.92 | 80.85 |
| jinaai/jina-reranker-v2-base-multilingual |  66.31   | 60.97 | 86.49 | 49.35 | 38.78 | 81.00 |    73.00    | 65.01 | 83.03 |  62.89  | 54.92 | 83.51 |
| jinaai/jina-colbert-v2                    |  66.73   | 61.47 | 85.78 | 49.59 | 39.20 | 79.86 |    73.24    | 65.36 | 82.96 |  63.19  | 55.34 | 82.87 |
| unicamp-dl/InRanker-base                  |  66.05   | 60.90 | 86.63 | 48.77 | 38.50 | 79.78 |    73.38    | 65.47 | 83.20 |  62.73  | 54.96 | 83.20 |
| rankGPT(Qwen/Qwen2-7B-Instruct)           |  63.11   | 58.26 | 77.91 | 49.50 | 39.06 | 75.90 |    70.13    | 62.31 | 79.11 |  60.91  | 53.21 | 77.64 |

Observations:
- Using a reranker can significantly improve the performance of the retrieval system.
- Cross-encoder-based rerankers have better performance than the other rerankers.
- BGE-reranker-M3 has the best performance on all datasets.
- rankGPT highly relies on the quality of the generator and has the highest overhead.

Conclusion:
We recommend using rerankers in latency insensitive scenarios. For building a prototype, we recommend using BGE-reranker-M3 or jina-reranker, which have the best performance on all datasets.

## Generator Benchmarks
> Experiment Settings: In these experiments, we employ the facebook/contriever-msmarco as our dense retriever. All the other settings are the same as the default settings in `ModularAssistant`. Specifically, for the Llama-3.3-70B-Instruct and Qwen2.5-72B-Instruct models, we deploy using Ollama with 4-bit quantization, while for the remaining models, we use VLLM for deployment and perform inference via the OpenAI API it provides.

| Methods                               | PopQA(%) |       |       | NQ(%) |       |       | TriviaQA(%) |       |       | Average |       |       |
| ------------------------------------- | :------: | :---: | :---: | :---: | :---: | :---: | :---------: | :---: | :---: | :-----: | :---: | :---: |
|                                       |    F1    |  EM   | Succ  |  F1   |  EM   | Succ  |     F1      |  EM   | Succ  |   F1    |  EM   | Succ  |
| Qwen/Qwen2-7B-Instruct \*             |  22.18   | 20.01 |   -   | 24.95 | 16.73 |   -   |    50.03    | 42.91 |   -   |  32.39  | 26.55 |   -   |
| Qwen/Qwen2-7B-Instruct                |  64.14   | 59.04 | 81.42 | 49.62 | 39.11 | 77.87 |    70.48    | 62.57 | 79.80 |  61.41  | 53.57 | 79.70 |
| Qwen/Qwen2.5-7B-Instruct \*           |  21.68   | 19.30 |   -   | 24.33 | 15.57 |   -   |    51.02    | 44.29 |   -   |  32.34  | 26.39 |   -   |
| Qwen/Qwen2.5-7B-Instruct              |  61.89   | 55.18 | 81.42 | 47.89 | 36.79 | 77.87 |    70.35    | 62.32 | 79.80 |  60.04  | 51.43 | 79.70 |
| Qwen/Qwen2.5-72B-Instruct \*          |   3.10   | 0.00  |   -   | 4.26  | 0.00  |   -   |    9.31     | 0.01  |   -   |  5.56   | 0.00  |   -   |
| Qwen/Qwen2.5-72B-Instruct             |  13.97   | 0.00  | 81.42 | 7.06  | 0.00  | 77.87 |    13.77    | 0.03  | 79.80 |  11.60  | 0.01  | 79.70 |
| meta-llama/Llama-3.1-8B-Instruct \*   |  22.08   | 19.44 |   -   | 34.33 | 23.41 |   -   |    64.46    | 56.52 |   -   |  40.29  | 33.12 |   -   |
| meta-llama/Llama-3.1-8B-Instruct      |  63.20   | 55.83 | 81.42 | 47.58 | 35.73 | 77.87 |    71.75    | 62.97 | 79.80 |  60.84  | 51.51 | 79.70 |
| meta-llama/Llama-3.3-70B-Instruct \*  |  30.14   | 27.81 |   -   | 46.40 | 32.30 |   -   |    79.60    | 72.04 |   -   |  52.05  | 44.05 |   -   |
| meta-llama/Llama-3.3-70B-Instruct     |  64.95   | 56.83 | 81.42 | 51.29 | 37.40 | 77.87 |    77.11    | 68.20 | 79.80 |  64.45  | 54.14 | 79.70 |
| mistralai/Mistral-7B-Instruct-v0.3 \* |  21.21   | 18.08 |   -   | 25.87 | 14.96 |   -   |    59.66    | 50.42 |   -   |  35.58  | 27.82 |   -   |
| mistralai/Mistral-7B-Instruct-v0.3    |  54.65   | 43.03 | 81.42 | 38.92 | 24.71 | 77.87 |    67.28    | 56.26 | 79.80 |  53.62  | 41.33 | 79.70 |
| nvidia/Llama3-ChatQA-2-8B \*          |  22.70   | 17.08 |   -   | 28.41 | 18.70 |   -   |    59.30    | 49.99 |   -   |  36.80  | 28.59 |   -   |
| nvidia/Llama3-ChatQA-2-8B             |  60.36   | 53.82 | 81.42 | 49.84 | 39.09 | 77.87 |    71.84    | 62.67 | 79.80 |  60.68  | 51.86 | 79.70 |

> \* stands for no retrieval results.

Observations:
- Without retrieval, models like Llama and Mistral have a relatively high response accuracy, and their performance continues to improve as the model size increases.
- With retrieval, Qwen2-7B-Instruct shows the most significant improvement.
- Qwen2.5 72B failed to follow the instruction to generate a brief answer and instead produced a longer response, resulting in a sharp decline in both F1 and EM scores. This may be related to the use of a quantized model.

