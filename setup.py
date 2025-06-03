import logging
import os
import re

import pybind11
from setuptools import Extension, setup

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


def get_version() -> str:
    with open(
        os.path.join("src", "flexrag", "utils", "default_vars.py"), encoding="utf-8"
    ) as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("__VERSION__")
        (version,) = re.findall(pattern, file_content)
        return version


def get_long_description() -> str:
    with open("README.md", encoding="utf-8") as f:
        return f.read()


setup(
    version=get_version(),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
)
