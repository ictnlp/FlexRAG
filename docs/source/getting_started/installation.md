# Installation
FlexRAG is a Python package that can be installed via `pip` or from source.

```{eval-rst}
.. important::
    FlexRAG requires Python 3.11 or later.
```

## Installation via `pip`
Before installing FlexRAG, ensure that `faiss` is installed on your environment. You can install it using the following command:

```bash
pip install faiss-cpu
```

```{eval-rst}
.. note::
    The pypi package is provided by the community.
    If you want to use the official `faiss` package or employ GPU for faster searching, you need to install it using `conda` and follow the instructions from its official `documentation <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>`_.
```

After installing `faiss`, run the following command to install FlexRAG:

```bash
pip install flexrag
```

## Installation from source
Alternatively, to install FlexRAG from the source, follow the steps below:
```bash
pip install pybind11 faiss-cpu

git clone https://github.com/ictnlp/FlexRAG.git
cd flexrag
pip install ./
```

```{eval-rst}
.. tip::
    You can also install the FlexRAG in *editable* mode with the `-e` flag.
    This allows you to make changes to the source code and have them reflected immediately without needing to reinstall the package.
```
