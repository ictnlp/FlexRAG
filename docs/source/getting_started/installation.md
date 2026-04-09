# Installation
FlexRAG is a Python package that can be installed via `pip` or from source.

```{important}
FlexRAG requires Python 3.11 or later.
```

## Installation via `pip`
The default installation includes `faiss-cpu`:

```bash
pip install flexrag
```

```{note}
`pip install flexrag` installs the community-maintained `faiss-cpu` wheel from PyPI by default.
If you need GPU acceleration, first install FlexRAG, then uninstall `faiss-cpu` and replace it with a GPU-enabled Faiss build by following the official [Faiss installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).
```

Optional capabilities are available as extras:

```bash
pip install "flexrag[ui]"
pip install "flexrag[web]"
pip install "flexrag[doc-parsers]"
```

## Installation from source
Alternatively, to install FlexRAG from the source, follow the steps below:
```bash
git clone https://github.com/ictnlp/FlexRAG.git
cd flexrag
pip install ./
```

If you need GPU-accelerated Faiss when installing from source, replace the default `faiss-cpu` dependency after installation by following the official [Faiss installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

To install optional capabilities from source, you can include the extras when
running `pip install`, for example `pip install ".[ui,web,doc-parsers]"`.

```{tip}
You can also install the FlexRAG in *editable* mode with the `-e` flag.
This allows you to make changes to the source code and have them reflected immediately without needing to reinstall the package.
```
