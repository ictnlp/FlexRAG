import os
from pathlib import Path

try:
    from ._version import version as __VERSION__
except ImportError:
    __VERSION__ = "0.0.0+unknown"
FLEXRAG_CACHE_DIR = Path(
    os.getenv("FLEXRAG_CACHE_DIR", Path.home() / ".cache" / "flexrag")
)
USER_MODULE_PATH = os.getenv("FLEXRAG_USER_MODULE_PATH", None)
