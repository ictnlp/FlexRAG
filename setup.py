import os
import re
import logging

import pybind11
from setuptools import Extension, find_packages, setup

logging.basicConfig(level=logging.INFO)


ext_modules = [
    Extension(
        "flexrag.metrics.lib_rel",
        ["src/flexrag/metrics/lib_rel.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
    ),
]


def get_requirements() -> list[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        requirements = [
            line.strip()
            for line in file_content.strip().split("\n")
            if not line.startswith("#")
        ]
    # as faiss may be installed using conda, we need to remove it from the requirements
    try:
        import faiss

        logging.info(f"Detected installed faiss: faiss {faiss.__version__}")
    except ImportError:
        requirements.append("faiss-cpu")
    return requirements


def get_version() -> str:
    with open(os.path.join("src", "flexrag", "utils.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("__VERSION__")
        (version,) = re.findall(pattern, file_content)
        return version


def get_long_description() -> str:
    with open("README.md", encoding="utf-8") as f:
        return f.read()


setup(
    name="flexrag",
    version=get_version(),
    author="Zhuocheng Zhang",
    author_email="zhuocheng_zhang@outlook.com",
    description="A RAG Framework for Information Retrieval and Generation.",
    url="https://github.com/ictnlp/flexrag",
    license="MIT License",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "flexrag": [
            "ranker/ranker_prompts/*.json",
            "assistant/assistant_prompts/*.json",
            "entrypoints/assets/*.png",
        ],
    },
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=get_requirements(),
    extras_require={
        "scann": ["scann>=1.3.2"],
        "annoy": ["annoy>1.17.0"],
        "llamacpp": ["llama_cpp_python>=0.2.84"],
        "minference": ["minference>=0.1.5"],
        "web": ["duckduckgo_search", "serpapi", "pyppeteer"],
        "docs": ["docling", "markitdown"],
        "all": [
            "llama_cpp_python>=0.2.84",
            "minference>=0.1.5",
            "PySocks>=1.7.1",
            "duckduckgo_search",
            "serpapi",
            "docling",
            "markitdown",
            "annoy>1.17.0",
        ],
        "dev": [
            "black",
            "pytest",
            "pytest-asyncio",
            "sphinx",
            "sphinx-autobuild",
            "myst-parser",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
)
