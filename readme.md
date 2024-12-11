<p align="center">
<img src="assets/librarian.png" width=55%>
</p>

![Language](https://img.shields.io/badge/language-python-brightgreen)
![github license](https://img.shields.io/github/license/ZhuochengZhang98/librarian)
[![DOI](https://zenodo.org/badge/900151663.svg)](https://doi.org/10.5281/zenodo.14306983)



Librarian is an open-source framework designed for Retrieval-Augmented Generation (RAG), combining state-of-the-art information retrieval techniques with large language models (LLMs) for enhanced, context-aware responses.

<p align="center">
<img src="assets/librarian-gui.gif" width=55%>
</p>

# :book: Table of Contents
- [:book: Table of Contents](#book-table-of-contents)
- [:sparkles: Key Features](#sparkles-key-features)
- [:rocket: Getting Started](#rocket-getting-started)
  - [Step 0. Installation](#step-0-installation)
    - [Install from pip](#install-from-pip)
    - [Install from source](#install-from-source)
  - [Step 1. Prepare the Retriever](#step-1-prepare-the-retriever)
    - [Download the Corpus](#download-the-corpus)
    - [Prepare the Index](#prepare-the-index)
  - [Step 2. Run your RAG Application](#step-2-run-your-rag-application)
    - [Run the Librarian Example RAG Application with GUI](#run-the-librarian-example-rag-application-with-gui)
    - [Run the Librarian Example Assistants for Knowledge Intensive Tasks](#run-the-librarian-example-assistants-for-knowledge-intensive-tasks)
    - [Build your own RAG Application](#build-your-own-rag-application)
- [:bar\_chart: Benchmarks](#bar_chart-benchmarks)
- [:label: License](#label-license)
- [:pen: Citation](#pen-citation)
- [:heart: Acknowledgements](#heart-acknowledgements)


# :sparkles: Key Features
- **Multiple Retriever Types**: Supports a wide range of retrievers, including sparse, dense, network-based, and multimodal retrievers.
- **Diverse Information Sources**: Allows integration of multiple data types, such as pure text, images, documents, web snapshots, and more.
- **Unified Configuration Management**: Leveraging OmegaConf and dataclasses, Librarian simplifies configuration and management across your entire project.
- **Out-of-the-Box Performance**: Comes with carefully optimized default configurations for retrievers, delivering solid performance without the need for extensive parameter tuning.
- **High Performance**: Built with persistent cache system and asynchronous methods to significantly improve speed and reduce latency in RAG workflows.
- **Research & Development Friendly**: Supports multiple development modes and includes a companion repository, [librarian_examples](https://github.com/ZhuochengZhang98/librarian_examples), to help you reproduce various RAG algorithms with ease.
- **Lightweight**: Designed with minimal overhead, Librarian is efficient and easy to integrate into your project.


# :rocket: Getting Started

## Step 0. Installation

### Install from pip
To install Librarian via pip:
```bash
pip install librarian-rag
```

### Install from source
Alternatively, to install from the source:
```bash
pip install pybind11

git clone https://github.com/ZhuochengZhang98/librarian.git
cd librarian
pip install ./
```
You can also install the Librarian in editable mode with the `-e` flag.


## Step 1. Prepare the Retriever

### Download the Corpus
Before starting you RAG application, you need to download the corpus. In this example, we will use the wikipedia corpus provided by [DPR](https://github.com/facebookresearch/DPR) as the corpus. You can download the corpus by running the following command:
```bash
# Download the corpus
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
# Unzip the corpus
gzip -d psgs_w100.tsv.gz
```

### Prepare the Index
After downloading the corpus, you need to build the index for the retriever. If you want to employ the dense retriever, you can simply run the following command to build the index:
```bash
CORPUS_PATH=psgs_w100.tsv.gz
CORPUS_FIELDS='[title,text]'
DB_PATH=<path_to_database>

python -m librarian.entrypoints.prepare_index \
    corpus_path=$CORPUS_PATH \
    saving_fields=$CORPUS_FIELDS \
    retriever_type=dense \
    dense_config.database_path=$DB_PATH \
    dense_config.encode_fields='[text]' \
    dense_config.passage_encoder_config.encoder_type=hf \
    dense_config.passage_encoder_config.hf_config.model_path='facebook/contriever' \
    dense_config.passage_encoder_config.hf_config.device_id=[0,1,2,3] \
    dense_config.index_type=faiss \
    dense_config.faiss_config.batch_size=4096 \
    dense_config.faiss_config.log_interval=100000 \
    dense_config.batch_size=4096 \
    dense_config.log_interval=100000 \
    reinit=True
```

If you want to employ the sparse retriever, you can run the following command to build the index:
```bash
CORPUS_PATH=psgs_w100.tsv.gz
CORPUS_FIELDS='[title,text]'
DB_PATH=<path_to_database>

python -m librarian.entrypoints.prepare_index \
    corpus_path=$CORPUS_PATH \
    saving_fields=$CORPUS_FIELDS \
    retriever_type=bm25s \
    bm25s_config.database_path=$DB_PATH \
    bm25s_config.indexed_fields='[title,text]' \
    bm25s_config.method=lucene \
    bm25s_config.batch_size=512 \
    bm25s_config.log_interval=100000 \
    reinit=True
```

## Step 2. Run your RAG Application
When the index is ready, you can run your RAG application. Here is an example of how to run a RAG application.

### Run the Librarian Example RAG Application with GUI
```bash
python -m librarian.entrypoints.run_interactive \
    assistant_type=modular \
    modular_config.used_fields=[title,text] \
    modular_config.retriever_type=dense \
    modular_config.dense_config.top_k=5 \
    modular_config.dense_config.database_path=${DB_PATH} \
    modular_config.dense_config.query_encoder_config.encoder_type=hf \
    modular_config.dense_config.query_encoder_config.hf_config.model_path='facebook/contriever' \
    modular_config.dense_config.query_encoder_config.hf_config.device_id=[0] \
    modular_config.response_type=short \
    modular_config.generator_type=openai \
    modular_config.openai_config.model_name='gpt-4o-mini' \
    modular_config.openai_config.api_key=$OPENAI_KEY \
    modular_config.do_sample=False
```

### Run the Librarian Example Assistants for Knowledge Intensive Tasks
You can evaluate your RAG application on several knowledge intensive datasets with great ease. The following command let you evaluate the modular RAG assistant with dense retriever on the Natural Questions (NQ) dataset:
```bash
OUTPUT_PATH=<path_to_output>
DB_PATH=<path_to_database>
OPENAI_KEY=<your_openai_key>

python -m librarian.entrypoints.run_assistant \
    data_path=flash_rag/nq/test.jsonl \
    output_path=${OUTPUT_PATH} \
    assistant_type=modular \
    modular_config.used_fields=[title,text] \
    modular_config.retriever_type=dense \
    modular_config.dense_config.top_k=10 \
    modular_config.dense_config.database_path=${DB_PATH} \
    modular_config.dense_config.query_encoder_config.encoder_type=hf \
    modular_config.dense_config.query_encoder_config.hf_config.model_path='facebook/contriever' \
    modular_config.dense_config.query_encoder_config.hf_config.device_id=[0] \
    modular_config.response_type=short \
    modular_config.generator_type=openai \
    modular_config.openai_config.model_name='gpt-4o-mini' \
    modular_config.openai_config.api_key=$OPENAI_KEY \
    modular_config.do_sample=False \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.context_preprocess.processor_type=[simplify_answer] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10
```

Similarly, you can evaluate the modular RAG assistant with sparse retriever on the Natural Questions dataset:
```bash
OUTPUT_PATH=<path_to_output>
DB_PATH=<path_to_database>
OPENAI_KEY=<your_openai_key>

python -m librarian.entrypoints.run_assistant \
    data_path=flash_rag/nq/test.jsonl \
    output_path=${OUTPUT_PATH} \
    assistant_type=modular \
    modular_config.used_fields=[title,text] \
    modular_config.retriever_type=bm25s \
    modular_config.bm25s_config.top_k=10 \
    modular_config.bm25s_config.database_path=${DB_PATH} \
    modular_config.response_type=short \
    modular_config.generator_type=openai \
    modular_config.openai_config.model_name='gpt-4o-mini' \
    modular_config.openai_config.api_key=$OPENAI_KEY \
    modular_config.do_sample=False \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.context_preprocess.processor_type=[simplify_answer] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10
```

You can also evaluate your own assistant by adding the `user_module=<your_module_path>` argument to the command.

### Build your own RAG Application
To build your own RAG application, you can create a new Python file and import the necessary Librarian modules. Here is an example of how to build a RAG application:
```python
from librarian.retrievers import DenseRetriever, DenseRetrieverConfig
from librarian.models import OpenAIGenerator, OpenAIGeneratorConfig
from librarian.prompt import ChatPrompt, ChatTurn


def main():
    # Initialize the retriever
    retriever = DenseRetriever(
        DenseRetrieverConfig(
            database_path='path_to_database',
            encode_fields=['text'],
            top_k=1,
            passage_encoder_config=HfEncoderConfig(
                encoder_type='hf',
                hf_config=HfConfig(
                    model_path='facebook/contriever',
                    device_id=[0],
                )
            ),
        )
    )

    # Initialize the generator
    generator = OpenAIGenerator(
        OpenAIGeneratorConfig(
            model_name='gpt-4o-mini',
            api_key='your_openai_key',
            do_sample=False
        )
    )

    # Run your RAG application
    Prompt = ChatPrompt()
    while True:
        query = input('Please input your query: ')
        context = retriever.retrieve(query)[0]
        prompt_str = f"Question: {query}\nContext: {context.data["text"]}"
        prompt.update(ChatTurn(role="user", content=prompt_str))
        response = generator.chat(prompt)
        prompt.update(ChatTurn(role="assistant", content=response))
        print(response)
    return

if __name__ == "__main__":
    main()
```
We also provide several detailed examples of how to build a RAG application in the [librarian examples](https://github.com/ZhuochengZhang98/librarian_examples) repository.


# :bar_chart: Benchmarks
We have conducted extensive benchmarks using the Librarian framework. For more details, please refer to the [benchmarks](benchmarks.md) page.

# :label: License
This repository is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.


# :pen: Citation
If you find this project helpful, please consider citing it:

```bibtex
@software{Librarian,
  author = {Zhang Zhuocheng},
  doi = {10.5281/zenodo.14306984},
  month = {12},
  title = {{Librarian}},
  url = {https://github.com/ZhuochengZhang98/librarian},
  version = {0.1.0},
  year = {2024}
}
```

# :heart: Acknowledgements
This project benefits from the following open-source projects:
- [Faiss](https://github.com/facebookresearch/faiss)
- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)
- [LanceDB](https://github.com/lancedb/lancedb)
- [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks)
