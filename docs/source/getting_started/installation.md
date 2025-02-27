# Installation
FlexRAG is a Python package that can be installed via `pip` or from source.

## Installation via `pip`
To install FlexRAG via pip, run the following command:

```bash
pip install flexrag
```

## Installation from source
Alternatively, to install FlexRAG from the source, follow the steps below:
```bash
pip install pybind11

git clone https://github.com/ictnlp/FlexRAG.git
cd flexrag
pip install ./
```
You can also install the FlexRAG in editable mode with the `-e` flag.

## Installation flags
FlexRAG can be installed with additional flags to enable specific features. The following flags are available:

| Flag       | pip install command             | Description                                         |
| ---------- | ------------------------------- | --------------------------------------------------- |
| scann      | pip install flexrag[scann]      | Install FlexRAG with the ScaNN index.               |
| annoy      | pip install flexrag[annoy]      | Install FlexRAG with the Annoy index.               |
| llamacpp   | pip install flexrag[llamacpp]   | Install FlexRAG with the LlamaCpp Generator.        |
| minference | pip install flexrag[minference] | Install FlexRAG with the Minference.                |
| web        | pip install flexrag[web]        | Install FlexRAG with the Web Retrievers.            |
| docs       | pip install flexrag[docs]       | Install FlexRAG with the Document Parser.           |
| all        | pip install flexrag[all]        | Install FlexRAG with most features.                 |
| dev        | pip install flexrag[dev]        | Install FlexRAG with the libraries for development. |
