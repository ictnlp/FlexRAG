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
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14306983.svg)](https://doi.org/10.5281/zenodo.14306983)

\[ [English](README.md) | [中文](README-zh.md) \]

FlexRAG 是一个具备高可复现性、易上手且性能优越的检索增强生成（Retrieval Augmented Generation, RAG）框架，专为科研与原型开发而设计。它支持**文本**、**多模态**以及**网络** RAG，提供完整的 RAG 流水线与评估流程，开箱即用，同时具备高效的**异步处理**与**持久化缓存**能力，助力快速搭建强大的 RAG 解决方案。

# 📖 目录
- [📖 目录](#-目录)
- [✨ 框架特色](#-框架特色)
- [📢 最新消息](#-最新消息)
- [🚀 框架入门](#-框架入门)
- [🏗️ FlexRAG 架构](#️-flexrag-架构)
- [📊 基准测试](#-基准测试)
- [🏷️ 许可证](#️-许可证)
- [🖋️ 引用](#️-引用)
- [❤️ 致谢](#️-致谢)


# ✨ 框架特色
- 🎯**高可复现性**：FlexRAG 附带了伴生仓库 [flexrag_examples](https://github.com/ictnlp/flexrag_examples)，为各类 RAG 算法提供详尽的复现案例；同时，在 HuggingFace Hub 上统一提供的检索器，确保您在相同环境下轻松复现实验结果。
- ✅**低上手难度**：一键下载并加载 HuggingFace Hub 上的检索器，免除了繁琐的构建流程；此外，FlexRAG 对默认配置进行了精心优化，使您在默认参数下就能获得出色性能，从而简化开发流程。
- 🌍**多样化应用场景**：FlexRAG 不仅适用于文本 RAG，还支持多模态及网络 RAG，为不同数据类型提供了广泛的应用可能。
- 🧪**科研优先**：为各类 RAG 任务提供统一评估流程，助您在不同数据集上轻松测试；同时，提供官方基准测试方便对比和查阅。
- ⚡**卓越性能**：利用持久化缓存和异步函数，助力高性能 RAG 开发。
- 🔄**全流程支持**：从文档信息提取、切分到检索与生成，再到生成质量评估，FlexRAG 完备支持 RAG 全生命周期的各个环节。
- 🛠️**模块化灵活设计**：采用轻量级模块化架构，FlexRAG 支持多种开发模式，助您快速构建专属 RAG 解决方案。

# 📢 最新消息
- **2025-03-24**: 中文文档上线啦！请访问 [文档](https://flexrag.readthedocs.io/zh-cn/latest/) 查看。
- **2025-02-25**: FlexRAG 的 LocalRetriever 现在支持从 [HuggingFace Hub](https://huggingface.co/collections/ICTNLP/flexrag-retrievers-67b5373b70123669108a2e59) 上加载啦！
- **2025-01-22**: 新的命令行入口 `run_retriever` 以及大量新的信息检索指标（如 `RetrievalMAP` ）现已上线，请阅读[文档](https://flexrag.readthedocs.io/en/latest/)以获取更多信息。
- **2025-01-08**: FlexRAG 现已支持 Windows 系统，您可以直接通过 `pip install flexrag` 来安装。
- **2025-01-08**: FlexRAG 在单跳QA数据集上的基准测试现已公开，详情请参考 [benchmarks](benchmarks/README.md) 页面。
- **2025-01-05**: FlexRAG 的[文档](https://flexrag.readthedocs.io/en/latest/)现已上线。

# 🚀 框架入门
从 `pip` 安装 FlexRAG:
```bash
pip install flexrag
```

访问我们的[文档](https://flexrag.readthedocs.io/zh-cn/latest/)以了解更多信息。
- [安装](https://flexrag.readthedocs.io/en/latest/getting_started/installation.html)
- [快速入门](https://flexrag.readthedocs.io/en/latest/getting_started/quickstart.html)
- [命令行入口](https://flexrag.readthedocs.io/en/latest/tutorial/entrypoints.html)

# 🏗️ FlexRAG 架构
FlexRAG 采用**模块化**架构设计，让您可以轻松定制和扩展框架以满足您的特定需求。下图说明了 FlexRAG 的架构：
<p align="center">
<img src="assets/Framework-FlexRAG-zh.png" width=70%>
</p>

# 📊 基准测试
我们利用 FlexRAG 进行了大量的基准测试，详情请参考 [benchmarks](benchmarks/README.md) 页面。

# 🏷️ 许可证
本仓库采用 **MIT License** 开源协议. 详情请参考 [LICENSE](LICENSE) 文件。

# 🖋️ 引用
如果您在研究中使用了 FlexRAG，请引用我们的项目：
```bibtex
@software{Zhang_FlexRAG_2025,
author = {Zhang, Zhuocheng and Feng, Yang and Zhang, Min},
doi = {10.5281/zenodo.14593327},
month = jan,
title = {{FlexRAG}},
url = {https://github.com/ictnlp/FlexRAG},
year = {2025}
}
```


# ❤️ 致谢
下面的开源项目对本项目有所帮助:
- [Faiss](https://github.com/facebookresearch/faiss)
- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)
- [LanceDB](https://github.com/lancedb/lancedb)
- [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks)
- [Chonkie](https://github.com/chonkie-ai/chonkie)
- [rerankers](https://github.com/AnswerDotAI/rerankers)
