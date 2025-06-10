import importlib
import os
import sys
from contextlib import contextmanager


@contextmanager
def set_env_var(key, value):
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is None:
            del os.environ[key]
        else:
            os.environ[key] = original_value


def load_user_module(module_path: str):
    module_path = os.path.abspath(module_path)
    module_parent, module_name = os.path.split(module_path)
    if module_name not in sys.modules:
        sys.path.insert(0, module_parent)
        importlib.import_module(module_name)
