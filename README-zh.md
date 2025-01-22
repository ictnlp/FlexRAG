<p align="center">
<img src="assets/flexrag-wide.png" width=55%>
</p>

![Language](https://img.shields.io/badge/language-python-brightgreen)
[![github license](https://img.shields.io/github/license/ictnlp/flexrag)](LICENSE)
[![Read the Docs](https://img.shields.io/readthedocs/flexrag)](https://flexrag.readthedocs.io/en/latest/)
[![PyPI - Version](https://img.shields.io/pypi/v/flexrag)](https://pypi.org/project/flexrag/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14593327.svg)](https://doi.org/10.5281/zenodo.14593327)

\[ [English](README.md) | [中文](README-zh.md) \]

FlexRAG 是一个灵活的高性能框架，专为检索增强生成 (RAG) 任务而设计。FlexRAG 支持多模态数据，提供统一的配置管理及开箱即用的检索系统，为科研和原型设计提供充分支持。

# 📖 目录
- [📖 目录](#-目录)
- [✨ 框架特色](#-框架特色)
- [📢 最新消息](#-最新消息)
- [🚀 框架入门](#-框架入门)
- [🏗️ FlexRAG 架构](#️-flexrag-架构)
- [📊 基准测试](#-基准测试)
- [🏷️ 许可证](#️-许可证)
- [❤️ 致谢](#️-致谢)


# ✨ 框架特色
- **多模态RAG**: FlexRAG 不仅限于基于文本的检索增强生成 (RAG)。它还支持多模态 RAG，为不同数据类型开辟了广泛的应用可能性。
- **多数据类型**: FlexRAG 支持多种数据格式，包括文本（例如 CSV、JSONL）、图像、文档、Web 快照等，让您可以灵活地处理各种数据源。
- **统一的配置管理**: 利用 python `dataclass` 和 [hydra-core](https://github.com/facebookresearch/hydra), FlexRAG 统一了配置管理，让 RAG 流程的配置变得更加简单。
- **开箱即用**: 通过精心优化的默认配置，FlexRAG 在默认配置下就有良好的性能，简化您的开发流程。
- **高性能**: 利用持久化缓存和异步函数，FlexRAG 显著提高了 RAG 流程的性能。
- **科研及开发友好**: 支持多种开发方式。此外，FlexRAG 提供了一个伴生仓库，[flexrag_examples](https://github.com/ictnlp/flexrag_examples)，来帮助您复现各类RAG算法。
- **轻量化**: FlexRAG 采用最少的开销设计，高效且易于集成到您的项目中。

# 📢 最新消息
- **2025-01-08**: FlexRAG 现已支持 Windows 系统，您可以直接通过 `pip install flexrag` 来安装。
- **2025-01-08**: FlexRAG 在单跳QA数据集上的基准测试现已公开，详情请参考 [benchmarks](benchmarks/README.md) 页面。
- **2025-01-05**: FlexRAG 的[文档](https://flexrag.readthedocs.io/en/latest/)现已上线。

# 🚀 框架入门
从 `pip` 安装 FlexRAG:
```bash
pip install flexrag
```

访问我们的[文档](https://flexrag.readthedocs.io/en/latest/)以了解更多信息。
- [安装](https://flexrag.readthedocs.io/en/latest/getting_started/installation.html)
- [快速入门](https://flexrag.readthedocs.io/en/latest/getting_started/quickstart.html)
- [命令行入口](https://flexrag.readthedocs.io/en/latest/tutorial/entrypoints.html)

# 🏗️ FlexRAG 架构
FlexRAG 采用**模块化**架构设计，让您可以轻松定制和扩展框架以满足您的特定需求。下图说明了 FlexRAG 的架构：
<p align="center">
<img src="assets/Framework-Librarian-v2.png" width=70%>
</p>

# 📊 基准测试
我们利用 FlexRAG 进行了大量的基准测试，详情请参考 [benchmarks](benchmarks/README.md) 页面。

# 🏷️ 许可证
本仓库采用 **MIT License** 开源协议. 详情请参考 [LICENSE](LICENSE) 文件。


<!-- # 🖋️ 引用
如果您觉得 FlexRAG 对您的研究有所帮助，请引用我们的工作:

```bibtex
@software{FlexRAG,
  author = {Zhang Zhuocheng},
  doi = {10.5281/zenodo.14306984},
  month = {12},
  title = {{FlexRAG}},
  url = {https://github.com/ictnlp/flexrag},
  version = {0.1.0},
  year = {2024}
}
``` -->

# ❤️ 致谢
下面的开源项目对本项目有所帮助:
- [Faiss](https://github.com/facebookresearch/faiss)
- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)
- [LanceDB](https://github.com/lancedb/lancedb)
- [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks)
- [Chonkie](https://github.com/chonkie-ai/chonkie)
- [rerankers](https://github.com/AnswerDotAI/rerankers)
