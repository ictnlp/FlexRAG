import os
import re
import logging

import pybind11
from setuptools import Extension, find_packages, setup

logging.basicConfig(level=logging.INFO)


ext_modules = [
    Extension(
        "librarian.metrics.lib_rel",
        ["src/librarian/metrics/lib_rel.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
    ),
]


def get_requirements() -> list[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [
            line.strip()
            for line in file_content.strip().split("\n")
            if not line.startswith("#")
        ]
    # as faiss may be installed using conda, we need to remove it from the requirements
    try:
        import faiss

        logging.info(f"Detected installed faiss: faiss {faiss.__version__}")
    except ImportError:
        pass
    else:
        lines = [line for line in lines if not line.startswith("faiss")]
    return lines


def get_version() -> str:
    with open(os.path.join("src", "librarian", "__init__.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("__VERSION__")
        (version,) = re.findall(pattern, file_content)
        return version


def get_long_description() -> str:
    with open("README.md", encoding="utf-8") as f:
        return f.read()


setup(
    name="librarian-rag",
    version=get_version(),
    author="Zhuocheng Zhang",
    author_email="zhuocheng_zhang@outlook.com",
    description="A RAG Framework for Information Retrieval and Generation.",
    url="https://github.com/ZhuochengZhang98/librarian",
    license="MIT License",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "librarian": [
            "ranker/ranker_prompts/*.json",
            "assistant/assistant_prompts/*.json",
        ],
    },
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=get_requirements(),
    extras_require={
        "gui": ["gradio>=5.8.0"],
        "scann": ["scann>=1.3.2"],
        "llamacpp": ["llama_cpp_python>=0.2.84"],
        "vllm": ["vllm>=0.5.0"],
        "ollama": ["ollama>=0.2.1"],
        "duckduckgo": ["duckduckgo_search>=6.1.6"],
        "anthropic": ["anthropic"],
        "minference": ["minference>=0.1.5"],
        "all": [
            "gradio>=5.8.0",
            "llama_cpp_python>=0.2.84",
            "vllm>=0.5.0",
            "ollama>=0.2.1",
            "duckduckgo_search>=6.1.6",
            "anthropic",
            "minference>=0.1.5",
            "PySocks>=1.7.1",
        ],
        "dev": [
            "black",
            "pytest",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
)
