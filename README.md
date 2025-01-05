<p align="center">
<img src="assets/flexrag-wide.png" width=55%>
</p>

![Language](https://img.shields.io/badge/language-python-brightgreen)
![github license](https://img.shields.io/github/license/ictnlp/flexrag)
<!-- [![DOI](https://zenodo.org/badge/900151663.svg)](https://doi.org/10.5281/zenodo.14306983) -->

\[ [English](README.md) | [中文](README-zh.md) \]

FlexRAG is a flexible and high-performance framework designed for Retrieval-Augmented Generation (RAG) tasks, offering support for multimodal data, seamless configuration management, and out-of-the-box performance for both research and prototyping.

https://github.com/user-attachments/assets/4dfc0ec9-686b-40e2-b1f0-daa2b918e093

# 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [✨ Key Features](#-key-features)
- [🚀 Getting Started](#-getting-started)
  - [Step 0. Installation](#step-0-installation)
    - [Install from pip](#install-from-pip)
    - [Install from source](#install-from-source)
  - [Step 1. Prepare the Retriever](#step-1-prepare-the-retriever)
    - [Download the Corpus](#download-the-corpus)
    - [Prepare the Index](#prepare-the-index)
  - [Step 2. Run FlexRAG Assistant](#step-2-run-flexrag-assistant)
    - [Run the FlexRAG Example RAG Application with GUI](#run-the-flexrag-example-rag-application-with-gui)
    - [Run the FlexRAG Example Assistants for Knowledge Intensive Tasks](#run-the-flexrag-example-assistants-for-knowledge-intensive-tasks)
    - [Build your own RAG Assistant](#build-your-own-rag-assistant)
    - [Run your own RAG Application](#run-your-own-rag-application)
- [🏗️ Architecture](#️-architecture)
- [📊 Benchmarks](#-benchmarks)
- [🏷️ License](#️-license)
- [❤️ Acknowledgements](#️-acknowledgements)


# ✨ Key Features
- **Multimodal RAG Support**: FlexRAG isn't limited to just text-based Retrieval-Augmented Generation (RAG). It also supports multimodal RAG, opening up a wide range of application possibilities across different media types.
- **Diverse Data Types**: FlexRAG enables seamless integration of multiple data formats, including text (e.g., CSV, JSONL), images, documents, web snapshots, and more, giving you flexibility in working with varied data sources.
- **Unified Configuration Management**: Leveraging python `dataclass` and [hydra-core](https://github.com/facebookresearch/hydra), FlexRAG simplifies configuration management, making it easier to handle complex setups and customize your workflow.
- **Out-of-the-Box**: With carefully optimized default configurations, FlexRAG delivers solid performance without the need for extensive parameter tuning.
- **High Performance**: Built with persistent cache system and asynchronous methods to significantly improve speed and reduce latency in RAG workflows.
- **Research & Development Friendly**: Supports multiple development modes and includes a companion repository, [flexrag_examples](https://github.com/ictnlp/flexrag_examples), to help you reproduce various RAG algorithms with ease.
- **Lightweight**: Designed with minimal overhead, FlexRAG is efficient and easy to integrate into your project.


# 🚀 Getting Started

## Step 0. Installation

### Install from pip
To install FlexRAG via pip:
```bash
pip install flexrag
```

### Install from source
Alternatively, to install from the source:
```bash
pip install pybind11

git clone https://github.com/ictnlp/flexrag.git
cd flexrag
pip install ./
```
You can also install the FlexRAG in editable mode with the `-e` flag.


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
CORPUS_PATH=[psgs_w100.tsv]
CORPUS_FIELDS='[title,text]'
DB_PATH=<path_to_database>

python -m flexrag.entrypoints.prepare_index \
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
CORPUS_PATH=[psgs_w100.tsv]
CORPUS_FIELDS='[title,text]'
DB_PATH=<path_to_database>

python -m flexrag.entrypoints.prepare_index \
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

## Step 2. Run FlexRAG Assistant
When the index is ready, you can run RAG `Assistant` provided by FlexRAG. Here is an example of how to run a `Modular Assistant`.

### Run the FlexRAG Example RAG Application with GUI
```bash
python -m flexrag.entrypoints.run_interactive \
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

### Run the FlexRAG Example Assistants for Knowledge Intensive Tasks
You can evaluate your RAG assistant on several knowledge intensive datasets with great ease. The following command let you evaluate the `Modular Assistant` with dense retriever on the Natural Questions (NQ) dataset:
```bash
OUTPUT_PATH=<path_to_output>
DB_PATH=<path_to_database>
OPENAI_KEY=<your_openai_key>

python -m flexrag.entrypoints.run_assistant \
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

Similarly, you can evaluate the `Modular Assistant` with sparse retriever on the Natural Questions dataset:
```bash
OUTPUT_PATH=<path_to_output>
DB_PATH=<path_to_database>
OPENAI_KEY=<your_openai_key>

python -m flexrag.entrypoints.run_assistant \
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

### Build your own RAG Assistant
To build your own RAG assistant, you can create a new Python file and import the necessary FlexRAG modules. Here is an example of how to build a RAG assistant:

```python
from dataclasses import dataclass

from flexrag.assistant import ASSISTANTS, AssistantBase
from flexrag.models import OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig


@dataclass
class SimpleAssistantConfig(DenseRetrieverConfig, OpenAIGeneratorConfig): ...


@ASSISTANTS("simple", config_class=SimpleAssistantConfig)
class SimpleAssistant(AssistantBase):
    def __init__(self, config: SimpleAssistantConfig):
        self.retriever = DenseRetriever(config)
        self.generator = OpenAIGenerator(config)
        return

    def answer(self, question: str) -> str:
        prompt = ChatPrompt()
        context = self.retriever.search(question)[0]
        prompt_str = ""
        for ctx in context:
            prompt_str += f"Question: {question}\nContext: {ctx.data['text']}"
        prompt.update(ChatTurn(role="user", content=prompt_str))
        response = self.generator.chat([prompt])[0][0]
        prompt.update(ChatTurn(role="assistant", content=response))
        return response
```
After defining the `SimpleAssistant` class and registering it with the `ASSISTANTS` decorator, you can run the assistant with the following command:
```bash
DB_PATH=<path_to_database>
OPENAI_KEY=<your_openai_key>
DATA_PATH=<path_to_data>
MODULE_PATH=<path_to_simple_assistant_module>

python -m flexrag.entrypoints.run_assistant \
    user_module=${MODULE_PATH} \
    data_path=${DATA_PATH} \
    assistant_type=simple \
    simple_config.model_name='gpt-4o-mini' \
    simple_config.api_key=${OPENAI_KEY} \
    simple_config.database_path=${DB_PATH} \
    simple_config.index_type=faiss \
    simple_config.query_encoder_config.encoder_type=hf \
    simple_config.query_encoder_config.hf_config.model_path='facebook/contriever' \
    simple_config.query_encoder_config.hf_config.device_id=[0] \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10
```
In [flexrag_examples](https://github.com/ictnlp/flexrag_examples) repository, we provide several detailed examples of how to build a RAG assistant.

### Run your own RAG Application
In addition to using FlexRAG's built-in Entrypoints to run your RAG Assistant, you can also use FlexRAG to build your own RAG application. The following is an example of how to build a RAG application.
```python
from flexrag.models import HFEncoderConfig, OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig


def main():
    # Initialize the retriever
    retriever_cfg = DenseRetrieverConfig(database_path="path_to_database", top_k=1)
    retriever_cfg.query_encoder_config.encoder_type = "hf"
    retriever_cfg.query_encoder_config.hf_config = HFEncoderConfig(
        model_path="facebook/contriever"
    )
    retriever = DenseRetriever(retriever_cfg)

    # Initialize the generator
    generator = OpenAIGenerator(
        OpenAIGeneratorConfig(
            model_name="gpt-4o-mini", api_key="your_openai_key", do_sample=False
        )
    )

    # Run your RAG application
    prompt = ChatPrompt()
    while True:
        query = input("Please input your query (type `exit` to exit): ")
        if query == "exit":
            break
        context = retriever.search(query)[0]
        prompt_str = ""
        for ctx in context:
            prompt_str += f"Question: {query}\nContext: {ctx.data['text']}"
        prompt.update(ChatTurn(role="user", content=prompt_str))
        response = generator.chat(prompt)
        prompt.update(ChatTurn(role="assistant", content=response))
        print(response)
    return


if __name__ == "__main__":
    main()
```
For more details on how to build your own RAG application, please refer to the [flexrag_examples](https://github.com/ictnlp/flexrag_examples) repository.


# 🏗️ Architecture
FlexRAG is designed with a **modular** architecture, allowing you to easily customize and extend the framework to meet your specific needs. The following diagram illustrates the architecture of FlexRAG:
<p align="center">
<img src="assets/Framework-Librarian-v2.png" width=70%>
</p>

# 📊 Benchmarks
We have conducted extensive benchmarks using the FlexRAG framework. For more details, please refer to the [benchmarks](benchmarks.md) page.

# 🏷️ License
This repository is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

# ❤️ Acknowledgements
This project benefits from the following open-source projects:
- [Faiss](https://github.com/facebookresearch/faiss)
- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)
- [LanceDB](https://github.com/lancedb/lancedb)
- [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks)
- [Chonkie](https://github.com/chonkie-ai/chonkie)
- [rerankers](https://github.com/AnswerDotAI/rerankers)
