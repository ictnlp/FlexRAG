<p align="center">
<img src="assets/flexrag-wide.png" width=55%>
</p>

![Language](https://img.shields.io/badge/language-python-brightgreen)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-blue)](https://pycqa.github.io/isort/)
[![github license](https://img.shields.io/github/license/ictnlp/FlexRAG)](LICENSE)
[![Read the Docs](https://img.shields.io/badge/docs-English-green)](https://flexrag.readthedocs.io/en/latest/)
[![Read the Docs](https://img.shields.io/badge/docs-Chinese-yellow)](https://flexrag.readthedocs.io/zh-cn/latest/)
[![PyPI - Version](https://img.shields.io/pypi/v/flexrag)](https://pypi.org/project/flexrag/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14593327.svg)](https://doi.org/10.5281/zenodo.14593327)

\[ [English](README.md) | [中文](README-zh.md) \]

FlexRAG is a highly reproducible, easy-to-use, and high-performance RAG framework designed for both research and application scenarios. It supports **text**, **multimodal**, and **web-based** RAG, providing a **complete RAG pipeline and evaluation process**. With built-in **asynchronous** processing and **persistent caching**, it ensures efficiency and scalability. Easily load retrievers from Hugging Face and quickly build powerful RAG solutions out of the box.

https://github.com/user-attachments/assets/4dfc0ec9-686b-40e2-b1f0-daa2b918e093

# 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [✨ Key Features](#-key-features)
- [📢 News](#-news)
- [🚀 Getting Started](#-getting-started)
- [🏗️ Architecture](#️-architecture)
- [📊 Benchmarks](#-benchmarks)
- [🏷️ License](#️-license)
- [❤️ Acknowledgements](#️-acknowledgements)


# ✨ Key Features
- 🎯 **High Reproducibility**: FlexRAG comes with a companion repository, [flexrag_examples](https://github.com/ictnlp/flexrag_examples), providing comprehensive reproduction cases for various RAG algorithms. Additionally, unified retrievers available on the HuggingFace Hub ensure easy replication of experimental results under the same environment.  
- ✅ **Low Learning Curve**: Download and load retrievers from the HuggingFace Hub with a single command, eliminating complex setup processes. Moreover, FlexRAG offers carefully optimized default configurations, allowing you to achieve excellent performance right out of the box and streamlining the development process.  
- 🌍 **Diverse Application Scenarios**: FlexRAG supports not only text-based RAG but also multimodal and network-based RAG, enabling broad applications across different data types.  
- 🧪 **Research-Oriented**: Provides a unified evaluation process for various RAG tasks, making it easy to test across different datasets. Official benchmark tests are also available for convenient comparison and reference.  
- ⚡ **Superior Performance**: Leverages persistent caching and asynchronous functions to enhance high-performance RAG development.  
- 🔄 **End-to-End Support**: From document information extraction and segmentation to retrieval, generation, and evaluation of output quality, FlexRAG fully supports every stage of the RAG lifecycle.  
- 🛠️ **Modular & Flexible Design**: With a lightweight modular architecture, FlexRAG accommodates multiple development modes, helping you quickly build customized RAG solutions.  

# 📢 News
- **2025-03-24**: The Chinese documentation is now available! Please visit the [documentation](https://flexrag.readthedocs.io/zh-cn/latest/) for more details.
- **2025-02-25**: FlexRAG's LocalRetriever now supports loading from the [HuggingFace Hub](https://huggingface.co/collections/ICTNLP/flexrag-retrievers-67b5373b70123669108a2e59).
- **2025-01-22**: A new entrypoint `run_retriever` and four new information retrieval metrics (e.g., `RetrievalMAP`) are now available. Check out the [documentation](https://flexrag.readthedocs.io/en/latest/) for more details.
- **2025-01-08**: We provide Windows wheels for FlexRAG. You can install FlexRAG via pip on Windows now.
- **2025-01-08**: The benchmark of FlexRAG on Single-hop QA tasks is now available. Check out the [benchmarks](benchmarks/README.md) for more details.
- **2025-01-05**: Documentation for FlexRAG is now available. Check out the [documentation](https://flexrag.readthedocs.io/en/latest/) for more details.

# 🚀 Getting Started
To install FlexRAG via pip:
```bash
pip install flexrag
```

Visit our [documentation](https://flexrag.readthedocs.io/en/latest/) to learn more.
- [Installation](https://flexrag.readthedocs.io/en/latest/getting_started/installation.html)
- [Quickstart](https://flexrag.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Entrypoints](https://flexrag.readthedocs.io/en/latest/tutorial/entrypoints.html)


# 🏗️ Architecture
FlexRAG is designed with a **modular** architecture, allowing you to easily customize and extend the framework to meet your specific needs. The following diagram illustrates the architecture of FlexRAG:
<p align="center">
<img src="assets/Framework-FlexRAG.png" width=70%>
</p>

# 📊 Benchmarks
We have conducted extensive benchmarks using the FlexRAG framework. For more details, please refer to the [benchmarks](benchmarks/README.md) page.

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
